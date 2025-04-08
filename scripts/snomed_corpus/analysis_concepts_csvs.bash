#! /bin/bash
# This script counts the number of lines in each CSV file in the current directory
wc -l *.csv|awk '{print  $2 ": " $1}'