#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, shutil
import pandas as pd
import numpy as np
from datetime import datetime


def create_conf_file(detector_name, output_name, uncert=0.5):

    if detector_name[0].lower() == "b": #BEGE
        siggen_dict = get_bege_detector_info(detector_name)

        imp_min, imp_max = -2, -5

    elif detector_name[0].lower() == "p": #ORTEC
        det_info = get_ortec_detector_info(detector_name)
        siggen_dict = detector_info_to_conf_file(det_info)

        #divide by 10 to go from 1E9 to 1E10
        imp_max = -(1+uncert)*np.amax((det_info["impurity_tail"], det_info["impurity_seed"]) ) / 10.
        imp_min = -(1+uncert)*np.amin((det_info["impurity_tail"], det_info["impurity_seed"]) ) / 10.

    else:
        raise ValueError("Unknown detector type for name {}".format(detector_name))

    with open(output_name, 'w') as conf_file:
        conf_file.write("# Auto-generated for {} from waffle at {}".format(detector_name, datetime.now()))

        for section_name, section_dict in siggen_dict.items():
            conf_file.write("\n[{}]\n".format(section_name))
            for k, v in section_dict.items():
                conf_file.write(format_conf_line(k,v))

    return imp_min, imp_max

def get_bege_detector_info(detector_name):
    bege_file_name = "beges.csv"

    cols = [0,3]
    col_names = ["detector_id", "operating_v"]

    df = pd.read_csv(bege_file_name, usecols=cols, names=col_names, index_col=0, skiprows=2)

    siggen_dict = get_bege_siggen_default(detector_name)
    siggen_dict["detector"]["xtal_HV"] = df.loc[int(detector_name[1:])]["operating_v"]

    return siggen_dict




def get_ortec_detector_info(detector_name):

    ortec_file_name = "ortec_ORTEC_Measurements.csv"
    starret_file_name = "ortec_starret_measurements.csv"
    contact_file_name = "ortec_starret_contacts.csv"

    ortec_cols = [0,1,2,6, 8,9,10,11,12,13,14]
    ortec_col_names = ["detector_id", "diameter", "length", "mass", "pc_diameter",
                        "pc_length", "dead_layer", "impurity_tail", "impurity_seed",
                        "depletion_v", "operating_v"
                        ]

    starret_cols = [0,6,7,8]
    starret_col_names = ["detector_id", "diameter", "length", "mass"]

    contact_cols = [0,2,3,4]

    df_ortec = pd.read_csv(ortec_file_name, usecols=ortec_cols, names=ortec_col_names, index_col=0, skiprows=1, na_values="**")
    df_starret = pd.read_csv(starret_file_name, usecols=starret_cols, names=starret_col_names,index_col=0, skiprows=2)
    df_contact = pd.read_csv(contact_file_name, usecols=contact_cols, index_col=0,header=0,)# dtype={'detector_id':str, 'height':float, 'dimple_bottom':float, 'dimple_diameter':float})

    #fill in missing deadlayer info
    df_ortec['dead_layer'].fillna((df_ortec['dead_layer'].mean()), inplace=True)

    try:
        detector_info = df_ortec.loc[detector_name]
    except KeyError as e:
        print(e)
        print("The detector {} you are generating was not located in the ortec measurement spreadsheet ({})".format(detector_name,ortec_file_name))
        exit()
    try:
        det_starret = df_starret.loc[detector_name]
    except KeyError as e:
        print(e)
        print("The detector {} you are generating was not located in the starrett measurement spreadsheet ({})".format(detector_name,starret_file_name))
        exit()
    try:
        det_contact = df_contact.loc[detector_name]
    except KeyError as e:
        print(e)
        print("The detector {} you are generating was not located in the contact measurement spreadsheet ({})".format(detector_name, contact_file_name))
        exit()

    for param_name in starret_col_names[1:]:
        if np.isnan(det_starret[param_name]):
            raise ValueError("Currently requiring there be starret measurements for every detector we use")

        if not np.isnan(det_starret[param_name]):
            detector_info[param_name] = det_starret[param_name]

    print("{}:".format(detector_name))
    print("  diameter: {}, {}".format(detector_info.pc_diameter, det_contact["dimple_diameter"] ))
    print("  length:   {}, {}".format( detector_info.pc_length, det_contact.height - det_contact.dimple_bottom))

    detector_info["pc_length"] = det_contact.height - det_contact.dimple_bottom
    detector_info["pc_diameter"] = det_contact["dimple_diameter"]

    return detector_info

def detector_info_to_conf_file(detector_info):

    #convert this all to siggen-style
    siggen_dict = get_ortec_siggen_default(detector_info.name)
    siggen_dict["geometry"]["xtal_length"] = np.round(detector_info["length"],2)
    siggen_dict["geometry"]["xtal_radius"] = np.round(detector_info["diameter"]/2.,2)
    siggen_dict["geometry"]["pc_length"] = np.round(detector_info["pc_length"],2)
    siggen_dict["geometry"]["pc_radius"] = np.round(detector_info["pc_diameter"]/2.,2)

    siggen_dict["detector"]["xtal_HV"] = detector_info["operating_v"]

    return siggen_dict

def get_ortec_siggen_default(detector_name):
    sig_dic = get_siggen_default(detector_name)

    sig_dic["geometry"]= {
        "top_bullet_radius":    1.2,
        "bottom_bullet_radius": 0,
        "bulletize_PC":    1,
        "taper_length": 4.5,
        "wrap_around_radius":0,
        "ditch_depth":0,
        "ditch_thickness":0
        }
    return sig_dic

def get_bege_siggen_default(detector_name):
    sig_dic = get_siggen_default(detector_name)

    sig_dic["geometry"]= {
        "xtal_length": 30.,
        "xtal_radius": 35.,
        "top_bullet_radius":    1.2,
        "bottom_bullet_radius": 1.2,
        "bulletize_PC":    0,
        "pc_length": 0.1,
        "pc_radius": 2.0,
        "taper_length": 0,
        "wrap_around_radius":11.1,
        "ditch_depth":2,
        "ditch_thickness":6.35
        }
    return sig_dic

def get_siggen_default(detector_name):
    return {
        "general":{
        "verbosity_level":0
        },
        "detector":{
        "field_name": "../fields/{}_ev.field".format(detector_name),
        "wp_name":    "../fields/{}_wpot.field".format(detector_name)
        },
        "siggen":{
        "time_steps_calc":8000,
        "step_time_calc":1.0,
        "step_time_out":1.0,
        }
        }

def format_conf_line(field_name, field_value):
    return "{:30}{}\n".format(field_name, field_value)
