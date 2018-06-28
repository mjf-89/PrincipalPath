#!/bin/bash

####
#Set the env variable PYTHONPATH to allow imports of principalpath module
#
#Usage:
#  source sourceme.sh
####
echo $PYTHONPATH | grep -q `pwd`

if [ $? -eq 1 ] ; then
	export PYTHONPATH=`pwd`:$PYTHONPATH
fi
