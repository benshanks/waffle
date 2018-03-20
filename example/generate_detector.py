#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, shutil

import waffle.detectors as wd

def main():

    names = ["P42664A", "P42665A"]

    for detector_name in names:

        conf_name = os.path.join(os.environ['DATADIR'], "siggen", "config_files", "{}.conf".format(detector_name))

        det_info = wd.get_detector_info(detector_name)
        wd.detector_info_to_conf_file(det_info, conf_name)
        wd.generate_fields(det_info, conf_name)

if __name__=="__main__":
    main()
