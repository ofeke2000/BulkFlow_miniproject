###################################################################################################
#
# settings.py 	        (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

"""
Global settings used across Colossus.
"""

###################################################################################################
# PATH TO COLOSSUS FILES
###################################################################################################

BASE_DIR = None
"""The directory in which Colossus stores files, e.g. cache files (without a backslash at the end). 
If ``None``, the home directory is used. If a home directory cannot be identified (an extremely 
rare case), the code directory is used."""

###################################################################################################
# PERSISTENT STORAGE
###################################################################################################

PERSISTENCE = 'rw'
"""This parameter determines whether Colossus stores persistent files such as the cosmology cache.
The parameter can take on any combination of read (``'r'``) and write (``'w'``), such as ``'rw'`` 
(read and write, the default), ``'r'`` (read only), ``'w'`` (write only), or ``''`` (no 
persistence). Note that this parameter is used as a default, but can still be changed for 
individual Colossus modules or objects, such as cosmology objects."""

PERSISTENCE_OLDEST_VERSION = '1.2.10'
"""If the colossus version that wrote a persistent quantitity was older than this version, the 
quantity is re-computed and the old file is removed. The version is increased whenever a numerical
change affects persistent quantities to ensure that no outdated data is kept."""
