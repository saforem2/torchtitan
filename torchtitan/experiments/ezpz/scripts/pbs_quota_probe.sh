#!/bin/bash
export PATH=/opt/pbs/bin:$PATH
# Held/queued chain links change constantly as dependencies release; only the
# RUNNING set and the count of pending links are worth waking anyone for.
run=$(qstat -u "$USER" 2>/dev/null | awk '/^[0-9]+\./ && $10=="R"{split($1,a,"."); printf "%s ", a[1]}')
[ -z "$run" ] && run="(none) "
pend=$(qstat -u "$USER" 2>/dev/null | awk '/^[0-9]+\./ && ($10=="Q"||$10=="H"){c++} END{print c+0}')
kb=$(lfs quota -p 2297 /lus/tegu 2>/dev/null | awk '/lus\/tegu/{v=$2; gsub(/\*/,"",v); print v}')
tb=$(awk -v k="${kb:-0}" 'BEGIN{printf "%.2f", k/1024/1024/1024}')
echo "RUNNING=${run}PENDING=$pend QUOTA_TB=$tb"
