#!/bin/bash

while read -r i o; do
    wget -O $o $i
done < urls.txt
