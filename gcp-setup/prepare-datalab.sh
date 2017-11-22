#!/bin/bash

#set environment variables
export PROJECT_ID=dswbiznesie
export ZONE_ID=europe-west1-b
export REGION_ID=europe-west1
export DISK_SIZE=20

#authenticate to google cloud
gcloud auth login

#spawn datalab vm
datalab create dswbiznesie --disk-size-gb $DISK_SIZE --zone $ZONE_ID

