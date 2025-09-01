-- AgentClient.client.lua (client-side, uses RemoteFunction RLStep)
-- Expects:
--   � Workspace/Checkpoints/CP_1, CP_2, ...
--   � ReplicatedStorage/RemoteFunction RLStep  (server calls Python)
--   � ServerScriptService/RLServer.lua (from previous message)

-- ========= CONFIG =========
local STEP_DT       = 0.05            -- Internal control/update interval (reward shaping, local sim)
local ACTION_DECISION_DT = 0.15       -- Server (HTTP) decision interval (reduce to avoid rate limits)
local CHECK_RADIUS  = 6               -- forgiving early on
local FALL_Y        = -20
local DOWN_RAY      = 50
local FWD_RAY       = 50
local VERBOSE       = true            -- master switch
local LOG_EVERY     = 10              -- steps
-- ==========================
-- Progress shaping / anti-loop config
local BACKTRACK_THRESH = 1.0          -- studs increase in distance considered a backtrack
local BACKTRACK_PENALTY_SCALE = 0.05  -- penalty per stud of backtrack beyond threshold
local STUCK_STEPS = 120               -- RL steps without sufficient improvement => stuck reset
local MIN_PROGRESS_EPS = 0.5          -- need at least this much dist improvement to count as progress
local STUCK_PENALTY = 8               -- penalty when flagged stuck (separate from normal -10 death)
local HEADING_SHAPE_SCALE = 0.002     -- reward scale for looking toward target (0 to disable)
local HORIZ_PROGRESS_SCALE = 0.08     -- additional reward for horizontal (XZ) approach
local JUMP_UP_BONUS = 0.5             -- bonus when upward velocity during approach & leaving ground
local IDLE_SPEED_THRESH = 1.0         -- below this horizontal speed counts as idle
local IDLE_PENALTY = 0.02             -- per step penalty while idle
local MAX_TIME_SINCE_JUMP = 5.0       -- cap for observation normalization
local BASE_WALK_SPEED = 16            -- baseline Humanoid WalkSpeed
local FORWARD_JUMP_SPEED = 38         -- target horizontal speed when initiating a forward jump
local JUMP_FORWARD_IMPULSE = 55       -- impulse strength for forward jump (scaled by mass)
local FACE_TARGET_DURING_AIR = false  -- keep facing next checkpoint while airborne to reduce spin (set true to force)
local AIR_STEER_KEEP_VEL = true       -- reserved flag for future in-air steering tweaks
-- Optional distinct starting spawn part (e.g. a Part or SpawnLocation) named START_SPAWN_NAME.
local START_SPAWN_NAME = "StartSpawn"
local function getStartSpawn()
	-- deep search so it can live inside a Folder
	return workspace:FindFirstChild(START_SPAWN_NAME, true)
end


local Players           = game:GetService("Players")
local RunService        = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

-- RemoteFunction provided by server (server makes the HTTP request)
local RLStep = ReplicatedStorage:WaitForChild("RLStep")

-- ----- tiny logger -----
local function now()
	return string.format("%.3f", os.clock())
end
local function v3(v)
	return string.format("(%.2f, %.2f, %.2f)", v.X, v.Y, v.Z)
end
local function log(fmt, ...)
	if VERBOSE then
		print(string.format("[%s] ", now()) .. string.format(fmt, ...))
	end
end
local function warnf(fmt, ...)
	warn(string.format("[%s] ", now()) .. string.format(fmt, ...))
end

-- ----- bootstrap character -----
local player = Players.LocalPlayer
if not player then
	warnf("No LocalPlayer (use Play, not Run)")
end
local char = player.Character or player.CharacterAdded:Wait()
local hrp  = char:WaitForChild("HumanoidRootPart")
local hum  = char:WaitForChild("Humanoid")

-- ----- checkpoints -----
local CHECKPOINT_FOLDER = workspace:FindFirstChild("Checkpoints")
assert(CHECKPOINT_FOLDER, "Workspace/Checkpoints folder is required with CP_1, CP_2, ...")

local function getCP(idx)
	return CHECKPOINT_FOLDER:FindFirstChild("CP_" .. tostring(idx))
end
local function getCPPos(idx)
	local p = getCP(idx)
	return p and p.Position or nil
end

-- ----- rays -----
local function rayDist(origin, dir, len)
	local params = RaycastParams.new()
	params.FilterDescendantsInstances = {char}
	params.FilterType = Enum.RaycastFilterType.Exclude
	local result = workspace:Raycast(origin, dir.Unit * len, params)
	if result then
		return (result.Position - origin).Magnitude
	else
		return len
	end
end

-- ----- obs/reward helpers -----
-- Allow an optional CP_0 (acts as first learning target). If absent we start at CP_1.
local HAS_CP0 = (CHECKPOINT_FOLDER:FindFirstChild("CP_0") ~= nil)
local INITIAL_CP_INDEX = HAS_CP0 and 0 or 1
local nextCP = INITIAL_CP_INDEX
local lastDist = math.huge
local died = false
local stepCounter = 0
local episodeCounter = 1
local lastObs = nil
local currentAction = 0               -- cached action applied every frame for smooth motion
local bestDistThisCP = math.huge      -- best (smallest) distance observed since current checkpoint became target
local noProgressSteps = 0             -- counter for stagnation
local timeSinceJump = 0
local wasGrounded = true

-- Convert action id -> desired move vector (world space forward/right relative to HRP)
local function actionToMove(action)
	if action == 1 then
		return hrp.CFrame.LookVector
	elseif action == 2 then
		return -hrp.CFrame.RightVector
	elseif action == 3 then
		return hrp.CFrame.RightVector
	elseif action == 5 then
		return hrp.CFrame.LookVector -- move forward + jump
	end
	return Vector3.zero
end

local function faceNextCheckpoint()
	if not FACE_TARGET_DURING_AIR then return end
	local cpPos = getCPPos(nextCP)
	if not cpPos then return end
	local pos = hrp.Position
	local target = Vector3.new(cpPos.X, pos.Y, cpPos.Z)
	local lookDir = target - pos
	if lookDir.Magnitude < 0.001 then return end
	hrp.CFrame = CFrame.new(pos, pos + lookDir.Unit)
end

local function applyForwardJumpBoost()
	local mass = hrp.AssemblyMass
	local forward = hrp.CFrame.LookVector
	local vel = hrp.AssemblyLinearVelocity
	local horiz = Vector3.new(vel.X, 0, vel.Z)
	local desired = forward * FORWARD_JUMP_SPEED
	local add = desired - horiz
	local impulse = Vector3.new(add.X, 0, add.Z) * mass
	hrp:ApplyImpulse(impulse)
end

hum.Died:Connect(function()
	died = true
	log("Humanoid.Died fired")
end)

-- Baseline movement tuning
hum.WalkSpeed = BASE_WALK_SPEED
hum.AutoRotate = true  -- allow natural rotation toward Move() direction

local function distanceToCP()
	local pos = getCPPos(nextCP)
	if not pos then return 0 end
	return (pos - hrp.Position).Magnitude
end

local function atCheckpoint()
	local pos = getCPPos(nextCP)
	if not pos then return false end
	return (hrp.Position - pos).Magnitude < CHECK_RADIUS
end

local function getObs()
	local pos = getCPPos(nextCP)
	local dx, dy, dz = 0, 0, 0
	if pos then
		local d = pos - hrp.Position
		dx, dy, dz = d.X, d.Y, d.Z
	end
	local v = hrp.AssemblyLinearVelocity
	local down = rayDist(hrp.Position, Vector3.new(0, -1, 0), DOWN_RAY)
	local forward = rayDist(hrp.Position, hrp.CFrame.LookVector, FWD_RAY)
	local grounded = (down < 5) and 1 or 0
	local cpDir = Vector3.new(dx, dy, dz)
	local look = hrp.CFrame.LookVector
	local angleCos = 0
	if cpDir.Magnitude > 0.001 then
		angleCos = look:Dot(cpDir.Unit)
	end
	local speedH = Vector3.new(v.X, 0, v.Z).Magnitude
	return {
		dx = dx, dy = dy, dz = dz,
		vx = v.X, vy = v.Y, vz = v.Z,
		down = down, forward = forward,
		angle = angleCos,
		grounded = grounded,
		speed = speedH,
		tJump = math.min(timeSinceJump, MAX_TIME_SINCE_JUMP)
	}
end

local function resetTo(cpIndex)
	-- Always prefer explicit start spawn if we haven't reached the first active checkpoint yet
	if cpIndex < INITIAL_CP_INDEX then
		local startPart = getStartSpawn()
		if startPart and startPart:IsA("BasePart") then
			hrp.CFrame = CFrame.new(startPart.Position + Vector3.new(0, 5, 0))
		else
			-- fallback: CP_1 if no start part exists
			local cp1 = CHECKPOINT_FOLDER:FindFirstChild("CP_1")
			if cp1 then
				hrp.CFrame = CFrame.new(cp1.Position + Vector3.new(0, 5, 0))
			end
		end
		hrp.AssemblyLinearVelocity = Vector3.zero
		hum:ChangeState(Enum.HumanoidStateType.Landed)
		return
	end

	-- normal case: go to last reached checkpoint
	local back = math.max(INITIAL_CP_INDEX, cpIndex)
	local cp = CHECKPOINT_FOLDER:FindFirstChild("CP_" .. tostring(back))
	if cp then
		hrp.CFrame = CFrame.new(cp.Position + Vector3.new(0, 5, 0))
		hrp.AssemblyLinearVelocity = Vector3.zero
		hum:ChangeState(Enum.HumanoidStateType.Landed)
	end
end


local function resetIfDeadOrFall()
	if died or hrp.Position.Y < FALL_Y then
		died = false
		resetTo(nextCP - 1)  -- this will use StartSpawn until CP_1 is reached
		return true
	end
	return false
end

-- ----- init -----
log("AgentClient starting. Character at %s", v3(hrp.Position))
if getCP(1) then
	log("Found CP_1 at %s", v3(getCP(1).Position))
else
	warnf("CP_1 not found under Workspace/Checkpoints")
end
lastObs = getObs()
lastDist = distanceToCP()
log("Initial nextCP=%d  dist=%.2f", nextCP, lastDist)

-- ----- main loop -----
local accum = 0
local actionAccum = 0                 -- accumulates time until next server decision
local pendingReward = 0              -- batched reward since last decision
-- High-frequency loop: apply last decided action every frame for smoothness
RunService.RenderStepped:Connect(function()
	-- apply cached movement continuously (Roblox internally blends)
	local moveVec = actionToMove(currentAction)
	hum:Move(moveVec, true)
	-- Only force facing if explicitly enabled
	if FACE_TARGET_DURING_AIR then
		faceNextCheckpoint()
	end
end)

-- Lower-frequency loop: query RL server & update action
RunService.Heartbeat:Connect(function(dt)
	timeSinceJump += dt
	accum += dt
	if accum < STEP_DT then return end
	accum = 0
	stepCounter += 1

	-- reward shaping (instantaneous)
	local reward = -0.01
	local dNow = distanceToCP()
	local shaped = 0
	local backtrackPenalty = 0
	local headingShape = 0
	local horizShape = 0
	local jumpBonus = 0
	local idlePenalty = 0

	-- forward progress shaping (positive)
	if dNow < lastDist then
		shaped = (lastDist - dNow) * 0.1
		reward += shaped
	end

	-- horizontal progress shaping (ignore vertical so climbing not penalized)
	local posCP = getCPPos(nextCP)
	if posCP then
		local toCP = posCP - hrp.Position
		local horizDist = Vector3.new(toCP.X, 0, toCP.Z).Magnitude
		local lastToCP = lastDist -- lastDist includes vertical; approximate horiz improvement using shaped when mostly horizontal
		-- Use previous and current horizontal via storing bestDistThisCP if needed (simpler: reward alignment * speed)
		local look = hrp.CFrame.LookVector
		horizShape = look:Dot(Vector3.new(toCP.X,0,toCP.Z).Unit) * (Vector3.new(hrp.AssemblyLinearVelocity.X,0,hrp.AssemblyLinearVelocity.Z).Magnitude) * 0.001 * HORIZ_PROGRESS_SCALE
		if horizShape > 0 then
			reward += horizShape
		end
	end

	-- backtrack penalty if moving away enough
	if dNow - lastDist > BACKTRACK_THRESH then
		backtrackPenalty = (dNow - lastDist) * BACKTRACK_PENALTY_SCALE
		reward -= backtrackPenalty
	end

	-- heading alignment shaping (encourage facing goal while moving)
	if HEADING_SHAPE_SCALE > 0 then
		local cpPos = getCPPos(nextCP)
		if cpPos then
			local dir = (cpPos - hrp.Position)
			if dir.Magnitude > 0.001 then
				local look = hrp.CFrame.LookVector
				headingShape = math.clamp(look:Dot(dir.Unit), -1, 1) * HEADING_SHAPE_SCALE
				reward += headingShape
			end
		end
	end

	-- jump encouragement: reward upward velocity right after leaving ground while moving toward target
	local vNow = hrp.AssemblyLinearVelocity
	local upward = vNow.Y > 2
	local currentGroundDist = rayDist(hrp.Position, Vector3.new(0,-1,0), DOWN_RAY)
	local currentGrounded = currentGroundDist < 5
	if (not currentGrounded) and wasGrounded and upward then
		jumpBonus = JUMP_UP_BONUS * math.clamp(headingShape*500, -1, 1)
		reward += jumpBonus
		timeSinceJump = 0
	end
	wasGrounded = currentGrounded

	-- idle penalty
	local speedH = Vector3.new(hrp.AssemblyLinearVelocity.X,0,hrp.AssemblyLinearVelocity.Z).Magnitude
	if speedH < IDLE_SPEED_THRESH then
		idlePenalty = IDLE_PENALTY
		reward -= idlePenalty
	end

	-- stagnation tracking
	if dNow + MIN_PROGRESS_EPS < bestDistThisCP then
		bestDistThisCP = dNow
		noProgressSteps = 0
	else
		noProgressSteps += 1
	end
	local forcedStuck = false
	if noProgressSteps >= STUCK_STEPS then
		-- treat as episode end due to being stuck (soft reset)
		reward -= STUCK_PENALTY
		forcedStuck = true
		noProgressSteps = 0
	end

	lastDist = dNow

	local reached = false
	if atCheckpoint() then
		reward += 10
		log("Reached CP_%d (dist=%.2f)  +10", nextCP, dNow)
		nextCP += 1
		reached = true
		hrp.AssemblyLinearVelocity = Vector3.zero
		-- reset progress trackers for new checkpoint target
		bestDistThisCP = math.huge
		noProgressSteps = 0
	end

	local done = resetIfDeadOrFall()
	if forcedStuck and not done then
		-- manually reset (without death/fall) to encourage exploration
		resetTo(nextCP - 1)
		done = true
		episodeCounter += 1
		lastDist = distanceToCP()
		log("Episode %d stuck-reset (-%d). New dist=%.2f nextCP=%d", episodeCounter, STUCK_PENALTY, lastDist, nextCP)
	elseif done then
		reward -= 10
		episodeCounter += 1
		lastDist = distanceToCP()
		log("Episode %d reset  (-10). New dist=%.2f  nextCP=%d", episodeCounter, lastDist, nextCP)
		-- reset progress trackers after respawn
		bestDistThisCP = math.huge
		noProgressSteps = 0
	end

	-- accumulate reward for batching
	pendingReward += reward
	actionAccum += STEP_DT

	if VERBOSE and (stepCounter % LOG_EVERY == 0 or reached) then
        log("step=%d ep=%d nextCP=%d pos=%s dist=%.2f instR=%.3f batchR=%.3f (prog=%.3f horiz=%.4f back=%.3f head=%.4f jump=%.3f idle=%.3f stuckSteps=%d chk=%s done=%s)",
            stepCounter, episodeCounter, nextCP, v3(hrp.Position), dNow, reward,
            pendingReward, shaped, horizShape, backtrackPenalty, headingShape, jumpBonus, idlePenalty,
            noProgressSteps, tostring(reached), tostring(done))
	end

	-- Decide whether to call server now (time or episode end)
	if actionAccum >= ACTION_DECISION_DT or done then
		-- ==== ask server (which calls Python) for action (batched reward) ====
		local action = 0
		local payload = { obs = lastObs, reward = pendingReward, done = done }
		local ok, result = pcall(function()
			return RLStep:InvokeServer(payload)   -- returns an action integer
		end)
		if ok then
			action = tonumber(result) or 0
		else
			warnf("RLStep InvokeServer failed: %s", tostring(result))
			action = 0
		end
		-- ====================================================

		-- cache action; jump immediate (one-shot). Movement handled per-frame.
		currentAction = action
		if action == 4 or action == 5 then
			hum.Jump = true
			if action == 5 then
				applyForwardJumpBoost()
			end
		end

		-- refresh obs after decision
		lastObs = getObs()
		pendingReward = 0
		actionAccum = 0
	end
end)
