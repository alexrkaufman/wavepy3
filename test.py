# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 14:45:43 2016

@author: Jeff
"""
# TODO make this more readable. Move the contents of Validate
# to this script. Make sure that this is well commented so people
# know what we are testing. Maybe provide multiple tests?
# potentially we should move these to their own folder for validation scripts
# for a few different facets of the package.
import WavePy3

X = WavePy3.WavePy3()

X.Validate(5)
