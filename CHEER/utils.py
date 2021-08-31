import os
import numpy as np 


# Order dict
order_dict = {0:"Bunyavirales", 1:"Mononegavirales", 2:"Nidovirales", 3:"Ortervirales", 4:"Picornavirales", 5:"Tymovirales"}

# Family dict
Bunyavirales_dict = {0:"Arenaviridae", 1:"Fimoviridae", 2:"Peribunyaviridae", 3:"Phenuiviridae", 4:"Tospoviridae"}

Mononegavirales_dict = {1:"Bornaviridae", 2:"Filoviridae", 3:"Paramyxoviridae", 4:"Rhabdoviridae"}

Nidovirales_dict = {1:"Arteriviridae", 2:"Coronaviridae", 3:"Mesoniviridae", 4:"Tobaniviridae"}

Ortervirales_dict = {1:"Caulimoviridae", 2:"Retroviridae"}

Picornavirales_dict = {1:"Dicistroviridae", 2:"Iflaviridae", 3:"Marnaviridae", 4:"Picornaviridae", 5:"Secoviridae"}

Tymovirales_dict = {1:"Alphaflexiviridae", 2:"Betaflexiviridae", 3:"Tymoviridae"}


# Genus dict
Arenaviridae_dict = {1:"Mammarenavirus", 2:"Reptarenavirus"}

Peribunyaviridae_dict = {1:"Herbevirus", 2:"Orthobunyavirus"}

Paramyxoviridae_dict = {1:"Henipavirus", 2:"Jeilongvirus", 3:"Morbillibirus", 4:"Pararubulavirus", 5:"Respirovirus"}

Rhabdoviridae_dict = {1:"Hapavirus", 2:"Ledantevirus", 3:"Lyssavirus", 4:"Vesivulovirus"}

Coronaviridae_dict = {1:"Betacoronavirus", 2:"Deltacoronavirus"}

Caulimoviridae_dict = {1:"Badnavirus", 2:"Caulimovirus", 3:"Cavemovirus"}

Retroviridae_dict = {1:"Alpharetrovirus", 2:"Deltaretrovirus", 3:"Epsilonretrovirus", 4:"Gammaretrovirus", 5:"Lentivirus", 6:"Simiispumavirus"}

Dicistroviridae_dict = {1:"Aparavirus", 2:"Cripavirus"}

Picornaviridae_dict = {1:"Cosavirus", 2:"Enterovirus", 3:"Hepatovirus", 4:"Kobuvirus", 5:"Parechovirus"}

Secoviridae_dict = {1:"Cheravirus", 2:"Comovirus", 3:"Fabavirus", 4:"Nepovirus", 5:"Torradovirus"}

Alphaflexiviridae_dict = {1:"Allexivirus", 2:"Potexvirus"}

Betaflexiviridae_dict = {1:"Capillovirus", 2:"Carlavirus", 3:"Divavirus", 4:"Foveavirus", 5:"Robigovirus", 6:"Trichovirus", 7:"Virivirus"}

Tymoviridae_dict = {1:"Marafivirus", 2:"Tymovirus"}


def get_leaf_num(name):
    if name == "Bunyavirales":
        return 5
    if name == "Mononegavirales":
        return 4
    if name == "Nidovirales":
        return 4
    if name == "Ortervirales":
        return 2
    if name == "Picornavirales":
        return 5
    if name == "Tymovirales":
        return 3
    if name == "Arenaviridae":
        return 2
    if name == "Peribunyaviridae":
        return 2
    if name == "Paramyxoviridae":
        return 5
    if name == "Rhabdoviridae":
        return 4
    if name == "Coronaviridae":
        return 2
    if name == "Caulimoviridae":
        return 3
    if name == "Retroviridae":
        return 6
    if name == "Dicistroviridae":
        return 2
    if name == "Picornaviridae":
        return 5
    if name == "Secoviridae":
        return 5
    if name == "Alphaflexiviridae":
        return 2
    if name == "Betaflexiviridae":
        return 7
    if name == "Tymoviridae":
        return 2
    else:
        return 0

def get_dict(name):
    if name == "Bunyavirales":
        return Bunyavirales_dict
    if name == "Mononegavirales":
        return Mononegavirales_dict
    if name == "Nidovirales":
        return Nidovirales_dict
    if name == "Ortervirales":
        return Ortervirales_dict
    if name == "Picornavirales":
        return Picornavirales_dict
    if name == "Tymovirales":
        return Tymovirales_dict
    if name == "Arenaviridae":
        return Arenaviridae_dict
    if name == "Peribunyaviridae":
        return Peribunyaviridae_dict
    if name == "Paramyxoviridae":
        return Paramyxoviridae_dict
    if name == "Rhabdoviridae":
        return Rhabdoviridae_dict
    if name == "Coronaviridae":
        return Coronaviridae_dict
    if name == "Caulimoviridae":
        return Caulimoviridae_dict
    if name == "Retroviridae":
        return Retroviridae_dict
    if name == "Dicistroviridae":
        return Dicistroviridae_dict
    if name == "Picornaviridae":
        return Picornaviridae_dict
    if name == "Secoviridae":
        return Secoviridae_dict
    if name == "Alphaflexiviridae":
        return Alphaflexiviridae_dict
    if name == "Betaflexiviridae":
        return Betaflexiviridae_dict
    if name == "Tymoviridae":
        return Tymoviridae_dict