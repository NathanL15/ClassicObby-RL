-- ServerScriptService/RLServer.lua
local HttpService = game:GetService("HttpService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local RL_URL = "http://127.0.0.1:5000/step" -- your Python server

local function postJSON(bodyTbl)
	local body = HttpService:JSONEncode(bodyTbl)
	local ok, res = pcall(function()
		return HttpService:PostAsync(RL_URL, body, Enum.HttpContentType.ApplicationJson)
	end)
	if not ok then
		warn("[RLServer] HTTP error:", res)
		return { action = 0 }
	end
	local ok2, decoded = pcall(function() return HttpService:JSONDecode(res) end)
	if not ok2 or type(decoded) ~= "table" then
		warn("[RLServer] Bad JSON reply")
		return { action = 0 }
	end
	return decoded
end

local rf = ReplicatedStorage:WaitForChild("RLStep")
rf.OnServerInvoke = function(player, payload)  -- payload = {obs=..., reward=..., done=...}
	-- (optional) per-player validation/logging here
	local reply = postJSON(payload)
	return reply.action or 0
end
