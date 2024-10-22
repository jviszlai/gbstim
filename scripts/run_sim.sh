#!/bin/bash

P_RANGE="0.001 0.003 0.005 0.007"
IDLE=-i # Include idle errors
T1=1000000 # us
T2=1000000 # us

python ler_sim.py codes/72-12-6-w-6.pkl -p $P_RANGE -t1 $T1 -t2 $T2 $IDLE
python ler_sim.py codes/90-8-10-w-6.pkl -p $P_RANGE -t1 $T1 -t2 $T2 $IDLE
python ler_sim.py codes/144-12-12-w-6.pkl -p $P_RANGE -t1 $T1 -t2 $T2 $IDLE
python ler_sim.py codes/128-16-8-w-8.pkl -p $P_RANGE -t1 $T1 -t2 $T2 $IDLE
python ler_sim.py codes/72-8-10-w-8.pkl -p $P_RANGE -t1 $T1 -t2 $T2 $IDLE
python ler_sim.py codes/96-10-12-w-8.pkl -p $P_RANGE -t1 $T1 -t2 $T2 $IDLE
