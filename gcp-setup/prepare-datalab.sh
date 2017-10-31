#!/bin/bash

#set environment variables
export PROJECT_ID=dswbiznesie
export ZONE_ID=europe-west1-b
export REGION_ID=europe-west1

#spawn datalab vm
datalab create dataengvm --disk-size-gb 20 --zone europe-west1-b

