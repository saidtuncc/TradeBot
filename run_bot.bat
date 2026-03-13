@echo off
title TradeBot v2.0
cd /d C:\TradeBot

:loop
echo [%date% %time%] Starting TradeBot...
python live_trader.py >> tradebot_output.log 2>&1
echo [%date% %time%] TradeBot crashed, restarting in 10 seconds...
timeout /t 10
goto loop
