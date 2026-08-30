#!/usr/bin/env bash
# End-to-end smoke test against a running backend.
# Usage: ./scripts/smoke.sh [base_url]
set -euo pipefail

BASE="${1:-http://127.0.0.1:8100}"

echo "== backend health =="
curl -sf "$BASE/" >/dev/null && echo "  ok: /"

echo "== create PvE game (3x3x3) =="
GID=$(curl -sf -X POST "$BASE/games" -H 'Content-Type: application/json' \
  -d '{"size":3,"mode":"pve","difficulty":"medium","x_agent":"model","o_agent":"model"}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['game_id'])")
echo "  game: $GID"
curl -sf "$BASE/games/$GID" | python3 -c "import sys,json;g=json.load(sys.stdin);print('  state ok: size',g['size'],'current',g['current_player'],'over',g['over'])"

echo "== streamed move (human + AI reply) =="
python3 - "$BASE" "$GID" <<'EOF'
import asyncio, json, sys, urllib.request
base, gid = sys.argv[1], sys.argv[2]

async def main():
    async with __import__("websockets").connect(f"ws://{base.split('://')[1]}/games/{gid}/stream") as ws:
        await ws.send(json.dumps({"type":"start"}))
        print("  ev:", json.loads(await asyncio.wait_for(ws.recv(), 5))["type"])
        req = urllib.request.Request(f"{base}/games/{gid}/move",
            data=json.dumps({"index":4}).encode(), headers={"Content-Type":"application/json"}, method="POST")
        print("  move ok:", json.loads(urllib.request.urlopen(req).read())["ok"])
        got_ai = False
        for _ in range(8):
            ev = json.loads(await asyncio.wait_for(ws.recv(), 5))
            if ev["type"] == "move" and ev["player"] == 2:
                got_ai = True
                print(f"  AI replied: index={ev['index']} coord={ev['coord']}")
                break
            print("  ev:", ev["type"], ev.get("player"))
        assert got_ai, "AI never replied!"
        print("  PvE flow PASSED")

asyncio.run(main())
EOF

echo "== AI vs AI game (3x3x3, model vs model) =="
GID2=$(curl -sf -X POST "$BASE/games" -H 'Content-Type: application/json' \
  -d '{"size":3,"mode":"ave","difficulty":"hard","x_agent":"model","o_agent":"model","ai_delay":0.05}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['game_id'])")
echo "  game: $GID2"
python3 - "$BASE" "$GID2" <<'EOF'
import asyncio, json, sys
base, gid = sys.argv[1], sys.argv[2]

async def main():
    async with __import__("websockets").connect(f"ws://{base.split('://')[1]}/games/{gid}/stream") as ws:
        await ws.send(json.dumps({"type":"start"}))
        moves, over = 0, False
        for _ in range(400):
            ev = json.loads(await asyncio.wait_for(ws.recv(), 10))
            if ev["type"] == "move":
                moves += 1
            if ev["type"] == "gameover":
                over = True
                print(f"  finished: winner={ev['game']['winner']} in {moves} moves")
                break
        assert over, "AI-vs-AI did not finish"
        print("  AI-vs-AI PASSED")

asyncio.run(main())
EOF

echo "ALL SMOKE TESTS PASSED"