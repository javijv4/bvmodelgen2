#!/bin/bash
for patient in AB-17 AS-10 AV-19 BI-18 CA-15 CW-21 DM-23 JL-3 JN-8 KL-4 KL-5 KR-13 MB-16 SL-16 TS-9 VB-1 ZS-11
do 
    echo -e ${patient}
    # python generate_uvc.py ${patient}
    python calculate_thickness.py ${patient}
    # python generate_lge_contours.py ${patient}
    # python rigid_mesh2lge_fit.py ${patient}
    # python lge_uvc_mesh_based.py ${patient}
    # python map_fibrosis.py ${patient}
done