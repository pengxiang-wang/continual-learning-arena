r"""
The submodule in `cl_datasets` for constants about datasets.
"""

__all__ = [
    "DatasetConstants",
    "ArabicHandwrittenDigitsConstants",
    "MNISTConstants",
    "EMNISTByClassConstants",
    "EMNISTByMergeConstants",
    "EMNISTBalancedConstants",
    "EMNISTLettersConstants",
    "EMNISTDigitsConstants",
    "FashionMNISTConstants",
    "KMNISTConstants",
    "NotMNISTConstants",
    "SignLanguageMNISTConstants",
    "KannadaMNISTConstants",
    "SVHNConstants",
    "CIFAR10Constants",
    "CIFAR100Constants",
    "TinyImageNetConstants",
    "CUB2002011Constants",
    "GTSRBConstants",
    "DATASET_CONSTANTS_MAPPING",
]

import tinyimagenet
import torch
import torchvision
from clarena.stl_datasets import raw
from torch.utils.data import Dataset


class DatasetConstants:
    r"""Base class for constants about datasets."""

    NUM_CLASSES: int
    r"""The number of classes in the dataset."""

    NUM_CHANNELS: int
    r"""The number of channels of images in the dataset."""

    IMG_SIZE: torch.Size
    r"""The size of images in the dataset (except channel)."""

    MEAN: tuple[float]
    r"""The mean values of each channel. """

    STD: tuple[float]
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, str | int]
    r"""The mapping from class index to original label name. """


class ArabicHandwrittenDigitsConstants(DatasetConstants):
    r"""Constants about Arabic Handwritten Digits dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in Arabic Handwritten Digits dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of SVHN images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of SVHN images (except channel)."""

    MEAN: tuple[float] = (0.3382,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2726,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # Arabic digit ٠ (0)
        1: 1,  # Arabic digit ١ (1)
        2: 2,  # Arabic digit ٢ (2)
        3: 3,  # Arabic digit ٣ (3)
        4: 4,  # Arabic digit ٤ (4)
        5: 5,  # Arabic digit ٥ (5)
        6: 6,  # Arabic digit ٦ (6)
        7: 7,  # Arabic digit ٧ (7)
        8: 8,  # Arabic digit ٨ (8)
        9: 9,  # Arabic digit ٩ (9)
    }
    r"""The mapping from class index to original label name. They correspond to the Arabic digits 0-9."""


class Caltech101Constants(DatasetConstants):
    r"""Constants about Caltech 101 dataset."""

    NUM_CLASSES: int = 101
    r"""The number of classes in Caltech 101 dataset."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Caltech 101 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Caltech 101 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the classes in Caltech 101 dataset."""


class Caltech256Constants(DatasetConstants):
    r"""Constants about Caltech 256 dataset."""

    NUM_CLASSES: int = 257
    r"""The number of classes in Caltech 256 dataset."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Caltech 256 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Caltech 256 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the classes in Caltech 256 dataset. Note that the class index 0 is reserved for the background class."""


class CelebAConstants(DatasetConstants):
    r"""Constants about CelebA dataset."""

    NUM_CLASSES: int = 10177
    r"""The number of classes in CelebA dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of CelebA images."""

    IMG_SIZE: torch.Size = torch.Size([178, 218])
    r"""The size of CelebA images (except channel)."""

    MEAN: tuple[float] = (0.5063, 0.4258, 0.3832)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2669, 0.2455, 0.241)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the identities of face images in CelebA dataset."""


class CIFAR10Constants(DatasetConstants):
    r"""Constants about CIFAR-10 dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in CIFAR-10 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of CIFAR images."""

    IMG_SIZE: torch.Size = torch.Size([32, 32])
    r"""The size of CIFAR images (except channel)."""

    MEAN: tuple[float] = (0.5074, 0.4867, 0.4411)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2011, 0.1987, 0.2025)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # airplane
        1: 1,  # automobile
        2: 2,  # bird
        3: 3,  # cat
        4: 4,  # deer
        5: 5,  # dog
        6: 6,  # frog
        7: 7,  # horse
        8: 8,  # ship
        9: 9,  # truck
    }
    r"""The mapping from class index to original label name. They correspond to the classes in CIFAR-10 dataset."""


class CIFAR100Constants(DatasetConstants):
    r"""Constants about CIFAR-100 dataset."""

    NUM_CLASSES: int = 100
    r"""The number of classes in CIFAR-100 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of CIFAR images."""

    IMG_SIZE: torch.Size = torch.Size([32, 32])
    r"""The size of CIFAR images (except channel)."""

    MEAN: tuple[float] = (0.5074, 0.4867, 0.4411)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2011, 0.1987, 0.2025)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # apple
        1: 1,  # aquarium fish
        2: 2,  # baby
        3: 3,  # bear
        4: 4,  # beaver
        5: 5,  # bed
        6: 6,  # bee
        7: 7,  # beetle
        8: 8,  # bicycle
        9: 9,  # bottle
        10: 10,  # bowl
        11: 11,  # boy
        12: 12,  # bridge
        13: 13,  # bus
        14: 14,  # butterfly
        15: 15,  # camel
        16: 16,  # can
        17: 17,  # castle
        18: 18,  # caterpillar
        19: 19,  # cattle
        20: 20,  # chair
        21: 21,  # chimpanzee
        22: 22,  # clock
        23: 23,  # cloud
        24: 24,  # couch
        25: 25,  # crab
        26: 26,  # crocodile
        27: 27,  # cup
        28: 28,  # dinosaur
        29: 29,  # dolphin
        30: 30,  # elephant
        31: 31,  # flatfish
        32: 32,  # forest
        33: 33,  # fox
        34: 34,  # girl
        35: 35,  # hamster
        36: 36,  # house
        37: 37,  # kangaroo
        38: 38,  # keyboard
        39: 39,  # lamp
        40: 40,  # lawn_mower
        41: 41,  # leopard
        42: 42,  # lion
        43: 43,  # lizard
        44: 44,  # lobster
        45: 45,  # man
        46: 46,  # maple_tree
        47: 47,  # motorcycle
        48: 48,  # mountain
        49: 49,  # mouse
        50: 50,  # mushroom
        51: 51,  # oak_tree
        52: 52,  # orange
        53: 53,  # orchid
        54: 54,  # otter
        55: 55,  # palm_tree
        56: 56,  # pear
        57: 57,  # pickup truck
        58: 58,  # pine tree
        59: 59,  # plain
        60: 60,  # plate
        61: 61,  # poppy
        62: 62,  # porcupine
        63: 63,  # possum
        64: 64,  # rabbit
        65: 65,  # raccoon
        66: 66,  # ray
        67: 67,  # road
        68: 68,  # rocket
        69: 69,  # rose
        70: 70,  # sea
        71: 71,  # seal
        72: 72,  # shark
        73: 73,  # shrew
        74: 74,  # skunk
        75: 75,  # skyscraper
        76: 76,  # snail
        77: 77,  # snake
        78: 78,  # spider
        79: 79,  # squirrel
        80: 80,  # streetcar
        81: 81,  # sunflower
        82: 82,  # sweet_pepper
        83: 83,  # table
        84: 84,  # tank
        85: 85,  # telephone
        86: 86,  # television
        87: 87,  # tiger
        88: 88,  # tractor
        89: 89,  # train
        90: 90,  # trout
        91: 91,  # tulip
        92: 92,  # turtle
        93: 93,  # wardrobe
        94: 94,  # whale
        95: 95,  # willow_tree
        96: 96,  # wolf
        97: 97,  # woman
        98: 98,  # worm
        99: 99,  # oak tree
    }
    r"""The mapping from class index to original label name. They correspond to the classes in CIFAR-100 dataset."""


class Country211Constants(DatasetConstants):
    r"""Constants about Country211 dataset."""

    NUM_CLASSES: int = 211
    r"""The number of classes in Country211 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Country211 images."""

    IMG_SIZE: torch.Size = torch.Size([256, 256])
    r"""The size of Country211 images (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Country211 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Country211 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the 211 countries in Country211 dataset."""


class CUB2002011Constants(DatasetConstants):
    r"""Constants about CUB-200-2011 dataset."""

    NUM_CLASSES: int = 200
    r"""The number of classes in CUB-200-2011 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of CUB-200-2011 images."""

    MEAN: tuple[float] = (0.4853, 0.4994, 0.4324)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2290, 0.2242, 0.2605)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # 001.Black_footed_Albatross
        1: 1,  # 002.Laysan_Albatross
        2: 2,  # 003.Sooty_Albatross
        3: 3,  # 004.Groove_billed_Ani
        4: 4,  # 005.Crested_Auklet
        5: 5,  # 006.Least_Auklet
        6: 6,  # 007.Parakeet_Auklet
        7: 7,  # 008.Rhinoceros_Auklet
        8: 8,  # 009.Brewer_Blackbird
        9: 9,  # 010.Red_winged_Blackbird
        10: 10,  # 011.Rusty_Blackbird
        11: 11,  # 012.Yellow_headed_Blackbird
        12: 12,  # 013.Bobolink
        13: 13,  # 014.Indigo_Bunting
        14: 14,  # 015.Lazuli_Bunting
        15: 15,  # 016.Painted_Bunting
        16: 16,  # 017.Cardinal
        17: 17,  # 018.Spotted_Catbird
        18: 18,  # 019.Gray_Catbird
        19: 19,  # 020.Yellow_breasted_Chat
        20: 20,  # 021.Eastern_Towhee
        21: 21,  # 022.Chuck_will_Widow
        22: 22,  # 023.Brandt_Cormorant
        23: 23,  # 024.Red_faced_Cormorant
        24: 24,  # 025.Pelagic_Cormorant
        25: 25,  # 026.Bronzed_Cowbird
        26: 26,  # 027.Shiny_Cowbird
        27: 27,  # 028.Brown_Creeper
        28: 28,  # 029.American_Crow
        29: 29,  # 030.Fish_Crow
        30: 30,  # 031.Black_billed_Cuckoo
        31: 31,  # 032.Mangrove_Cuckoo
        32: 32,  # 033.Yellow_billed_Cuckoo
        33: 33,  # 034.Gray_crowned_Rosy_Finch
        34: 34,  # 035.Purple_Finch
        35: 35,  # 036.Northern_Flicker
        36: 36,  # 037.Acadian_Flycatcher
        37: 37,  # 038.Great_Crested_Flycatcher
        38: 38,  # 039.Least_Flycatcher
        39: 39,  # 040.Olive_sided_Flycatcher
        40: 40,  # 041.Scissor_tailed_Flycatcher
        41: 41,  # 042.Vermilion_Flycatcher
        42: 42,  # 043.Yellow_bellied_Flycatcher
        43: 43,  # 044.Frigatebird
        44: 44,  # 045.Northern_Fulmar
        45: 45,  # 046.Gadwall
        46: 46,  # 047.American_Goldfinch
        47: 47,  # 048.European_Goldfinch
        48: 48,  # 049.Boat_tailed_Grackle
        49: 49,  # 050.Eared_Grebe
        50: 50,  # 051.Horned_Grebe
        51: 51,  # 052.Pied_billed_Grebe
        52: 52,  # 053.Western_Grebe
        53: 53,  # 054.Blue_Grosbeak
        54: 54,  # 055.Evening_Grosbeak
        55: 55,  # 056.Pine_Grosbeak
        56: 56,  # 057.Rose_breasted_Grosbeak
        57: 57,  # 058.Pigeon_Guillemot
        58: 58,  # 059.California_Gull
        59: 59,  # 060.Glaucous_winged_Gull
        60: 60,  # 061.Heermann_Gull
        61: 61,  # 062.Herring_Gull
        62: 62,  # 063.Ivory_Gull
        63: 63,  # 064.Ring_billed_Gull
        64: 64,  # 065.Slaty_backed_Gull
        65: 65,  # 066.Western_Gull
        66: 66,  # 067.Anna_Hummingbird
        67: 67,  # 068.Ruby_throated_Hummingbird
        68: 68,  # 069.Rufous_Hummingbird
        69: 69,  # 070.Green_Violetear
        70: 70,  # 071.Long_tailed_Jaeger
        71: 71,  # 072.Pomarine_Jaeger
        72: 72,  # 073.Blue_Jay
        73: 73,  # 074.Florida_Jay
        74: 74,  # 075.Green_Jay
        75: 75,  # 076.Dark_eyed_Junco
        76: 76,  # 077.Tropical_Kingbird
        77: 77,  # 078.Gray_Kingbird
        78: 78,  # 079.Belted_Kingfisher
        79: 79,  # 080.Green_Kingfisher
        80: 80,  # 081.Pied_Kingfisher
        81: 81,  # 082.Ringed_Kingfisher
        82: 82,  # 083.White_breasted_Kingfisher
        83: 83,  # 084.Red_legged_Kittiwake
        84: 84,  # 085.Horned_Lark
        85: 85,  # 086.Pacific_Loon
        86: 86,  # 087.Mallard
        87: 87,  # 088.Western_Meadowlark
        88: 88,  # 089.Hooded_Merganser
        89: 89,  # 090.Red_breasted_Merganser
        90: 90,  # 091.Mockingbird
        91: 91,  # 092.Nighthawk
        92: 92,  # 093.Clark_Nutcracker
        93: 93,  # 094.White_breasted_Nuthatch
        94: 94,  # 095.Baltimore_Oriole
        95: 95,  # 096.Hooded_Oriole
        96: 96,  # 097.Orchard_Oriole
        97: 97,  # 098.Scott_Oriole
        98: 98,  # 099.Ovenbird
        99: 99,  # 100.Brown_Pelican
        100: 100,  # 101.White_Pelican
        101: 101,  # 102.Western_Wood_Pewee
        102: 102,  # 103.Sayornis
        103: 103,  # 104.American_Pipit
        104: 104,  # 105.Whip_poor_Will
        105: 105,  # 106.Horned_Puffin
        106: 106,  # 107.Common_Raven
        107: 107,  # 108.White_necked_Raven
        108: 108,  # 109.American_Redstart
        109: 109,  # 110.Geococcyx
        110: 110,  # 111.Loggerhead_Shrike
        111: 111,  # 112.Great_Grey_Shrike
        112: 112,  # 113.Baird_Sparrow
        113: 113,  # 114.Black_throated_Sparrow
        114: 114,  # 115.Brewer_Sparrow
        115: 115,  # 116.Chipping_Sparrow
        116: 116,  # 117.Clay_colored_Sparrow
        117: 117,  # 118.House_Sparrow
        118: 118,  # 119.Field_Sparrow
        119: 119,  # 120.Fox_Sparrow
        120: 120,  # 121.Grasshopper_Sparrow
        121: 121,  # 122.Harris_Sparrow
        122: 122,  # 123.Henslow_Sparrow
        123: 123,  # 124.Le_Conte_Sparrow
        124: 124,  # 125.Lincoln_Sparrow
        125: 125,  # 126.Nelson_Sharp_tailed_Sparrow
        126: 126,  # 127.Savannah_Sparrow
        127: 127,  # 128.Seaside_Sparrow
        128: 128,  # 129.Song_Sparrow
        129: 129,  # 130.Swainson_Sparrow
        130: 130,  # 131.Tree_Sparrow
        131: 131,  # 132.Vesper_Sparrow
        132: 132,  # 133.White_crowned_Sparrow
        133: 133,  # 134.White_throated_Sparrow
        134: 134,  # 135.Bank_Swallow
        135: 135,  # 136.Barn_Swallow
        136: 136,  # 137.Cliff_Swallow
        137: 137,  # 138.Tree_Swallow
        138: 138,  # 139.Scarlet_Tanager
        139: 139,  # 140.Summer_Tanager
        140: 140,  # 141.Artic_Tern
        141: 141,  # 142.Black_Tern
        142: 142,  # 143.Caspian_Tern
        143: 143,  # 144.Common_Tern
        144: 144,  # 145.Elegant_Tern
        145: 145,  # 146.Forsters_Tern
        146: 146,  # 147.Least_Tern
        147: 147,  # 148.Green_tailed_Towhee
        148: 148,  # 149.Brown_Thrasher
        149: 149,  # 150.Sage_Thrasher
        150: 150,  # 151.Black_capped_Vireo
        151: 151,  # 152.Blue_headed_Vireo
        152: 152,  # 153.Philadelphia_Vireo
        153: 153,  # 154.Red_eyed_Vireo
        154: 154,  # 155.Warbling_Vireo
        155: 155,  # 156.White_eyed_Vireo
        156: 156,  # 157.Yellow_throated_Vireo
        157: 157,  # 158.Bay_breasted_Warbler
        158: 158,  # 159.Black_and_white_Warbler
        159: 159,  # 160.Black_throated_Blue_Warbler
        160: 160,  # 161.Black_throated_Green_Warbler
        161: 161,  # 162.Blue_winged_Warbler
        162: 162,  # 163.Canada_Warbler
        163: 163,  # 164.Cape_May_Warbler
        164: 164,  # 165.Cerulean_Warbler
        165: 165,  # 166.Chestnut_sided_Warbler
        166: 166,  # 167.Golden_winged_Warbler
        167: 167,  # 168.Hooded_Warbler
        168: 168,  # 169.Kentucky_Warbler
        169: 169,  # 170.Magnolia_Warbler
        170: 170,  # 171.Mourning_Warbler
        171: 171,  # 172.Myrtle_Warbler
        172: 172,  # 173.Nashville_Warbler
        173: 173,  # 174.Orange_crowned_Warbler
        174: 174,  # 175.Palm_Warbler
        175: 175,  # 176.Pine_Warbler
        176: 176,  # 177.Prairie_Warbler
        177: 177,  # 178.Prothonotary_Warbler
        178: 178,  # 179.Swainson_Warbler
        179: 179,  # 180.Tennessee_Warbler
        180: 180,  # 181.Wilson_Warbler
        181: 181,  # 182.Worm_eating_Warbler
        182: 182,  # 183.Yellow_Warbler
        183: 183,  # 184.Northern_Waterthrush
        184: 184,  # 185.Louisiana_Waterthrush
        185: 185,  # 186.Boogie_Waxwing
        186: 186,  # 187.Cedar_Waxwing
        187: 187,  # 188.American_Three_toed_Woodpecker
        188: 188,  # 189.Downy_Woodpecker
        189: 189,  # 190.Hairy_Woodpecker
        190: 190,  # 191.Northern_Flicker
        191: 191,  # 192.Pileated_Woodpecker
        192: 192,  # 193.Red_bellied_Woodpecker
        193: 193,  # 194.Red_cockaded_Woodpecker
        194: 194,  # 195.Red_headed_Woodpecker
        195: 195,  # 196.White_headed_Woodpecker
        196: 196,  # 197.Wilson_Snipe
        197: 197,  # 198.Wilson_Warbler
        198: 198,  # 199.Winter_Wren
        199: 199,  # 200.Yellow_bellied_Sapsucker
    }
    r"""The mapping from class index to original label name. They correspond to the classes in CUB-200-2011 dataset."""


class DTDConstants(DatasetConstants):
    r"""Constants about DTD dataset."""

    NUM_CLASSES: int = 47
    r"""The number of classes in DTD dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of DTD images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Country211 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Country211 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # bark
        1: 1,  # blotchy
        2: 2,  # bricks
        3: 3,  # cork
        4: 4,  # fabric
        5: 5,  # feathers
        6: 6,  # flower
        7: 7,  # foliage
        8: 8,  # food
        9: 9,  # fur
        10: 10,  # grass
        11: 11,  # ground
        12: 12,  # hair
        13: 13,  # hexagons
        14: 14,  # jagged
        15: 15,  # leaves
        16: 16,  # linen
        17: 17,  # marble
        18: 18,  # mesh
        19: 19,  # mosaic
        20: 20,  # mountain
        21: 21,  # paint
        22: 22,  # paper
        23: 23,  # pebbles
        24: 24,  # planks
        25: 25,  # polka dots
        26: 26,  # ripples
        27: 27,  # sand
        28: 28,  # sea
        29: 29,  # shells
        30: 30,  # shiny
        31: 31,  # sparse
        32: 32,  # spotted
        33: 33,  # stripes
        34: 34,  # swirl
        35: 35,  # tiles
        36: 36,  # tree rings
        37: 37,  # water
        38: 38,  # web
        39: 39,  # wood
        40: 40,  # zebra
        41: 41,  # frosted
        42: 42,  # gravel
        43: 43,  # spikes
        44: 44,  # leather
        45: 45,  # woven
        46: 46,  # terrazzo
    }
    r"""The mapping from class index to original label name. They correspond to the texture classes in DTD dataset."""


class EMNISTByClassConstants(DatasetConstants):
    r"""Constants about EMNIST ByClass dataset."""

    NUM_CLASSES: int = 62
    r"""The number of classes in EMNIST ByClass dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of EMNIST ByClass images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of EMNIST ByClass images (except channel)."""

    MEAN: tuple[float] = (0.1751,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3332,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # digit 0
        1: 1,  # digit 1
        2: 2,  # digit 2
        3: 3,  # digit 3
        4: 4,  # digit 4
        5: 5,  # digit 5
        6: 6,  # digit 6
        7: 7,  # digit 7
        8: 8,  # digit 8
        9: 9,  # digit 9
        10: 10,  # letter A
        11: 11,  # letter B
        12: 12,  # letter C
        13: 13,  # letter D
        14: 14,  # letter E
        15: 15,  # letter F
        16: 16,  # letter G
        17: 17,  # letter H
        18: 18,  # letter I
        19: 19,  # letter J
        20: 20,  # letter K
        21: 21,  # letter L
        22: 22,  # letter M
        23: 23,  # letter N
        24: 24,  # letter O
        25: 25,  # letter P
        26: 26,  # letter Q
        27: 27,  # letter R
        28: 28,  # letter S
        29: 29,  # letter T
        30: 30,  # letter U
        31: 31,  # letter V
        32: 32,  # letter W
        33: 33,  # letter X
        34: 34,  # letter Y
        35: 35,  # letter Z
        36: 36,  # letter a
        37: 37,  # letter b
        38: 38,  # letter c
        39: 39,  # letter d
        40: 40,  # letter e
        41: 41,  # letter f
        42: 42,  # letter g
        43: 43,  # letter h
        44: 44,  # letter i
        45: 45,  # letter j
        46: 46,  # letter k
        47: 47,  # letter l
        48: 48,  # letter m
        49: 49,  # letter n
        50: 50,  # letter o
        51: 51,  # letter p
        52: 52,  # letter q
        53: 53,  # letter r
        54: 54,  # letter s
        55: 55,  # letter t
        56: 56,  # letter u
        57: 57,  # letter v
        58: 58,  # letter w
        59: 59,  # letter x
        60: 60,  # letter y
        61: 61,  # letter z
    }
    r"""The mapping from class index to original label name. They correspond to the digits 0-9 and letters A-Z, a-z."""


class EMNISTByMergeConstants(DatasetConstants):
    r"""Constants about EMNIST ByMerge dataset."""

    NUM_CLASSES: int = 47
    r"""The number of classes in EMNIST ByMerge dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of EMNIST ByMerge images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of EMNIST ByMerge images (except channel)."""

    MEAN: tuple[float] = (0.1740,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3310,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # digit 0
        1: 1,  # digit 1
        2: 2,  # digit 2
        3: 3,  # digit 3
        4: 4,  # digit 4
        5: 5,  # digit 5
        6: 6,  # digit 6
        7: 7,  # digit 7
        8: 8,  # digit 8
        9: 9,  # digit 9
        10: 10,  # letter A/a
        11: 11,  # letter B/b
        12: 12,  # letter C/c
        13: 13,  # letter D/d
        14: 14,  # letter E/e
        15: 15,  # letter F/f
        16: 16,  # letter G/g
        17: 17,  # letter H/h
        18: 18,  # letter I/i
        19: 19,  # letter J/j
        20: 20,  # letter K/k
        21: 21,  # letter L/l
        22: 22,  # letter M/m
        23: 23,  # letter N/n
        24: 24,  # letter O/o
        25: 25,  # letter P/p
        26: 26,  # letter Q/q
        27: 27,  # letter R/r
        28: 28,  # letter S/s
        29: 29,  # letter T/t
        30: 30,  # letter U/u
        31: 31,  # letter V/v
        32: 32,  # letter W/w
        33: 33,  # letter X/x
        34: 34,  # letter Y/y
        35: 35,  # letter Z/z
        36: 36,  # letter a
        37: 37,  # letter b
        38: 38,  # letter d
        39: 39,  # letter e
        40: 40,  # letter f
        41: 41,  # letter g
        42: 42,  # letter h
        43: 43,  # letter n
        44: 44,  # letter q
        45: 45,  # letter r
        46: 46,  # letter t
    }
    r"""The mapping from class index to original label name. They correspond to the digits 0-9 and letters A-Z, a-z. The letters a, b, d, e, f, g, h, n, q, r, t are merged with their capital letters."""


class EMNISTBalancedConstants(DatasetConstants):
    r"""Constants about EMNIST Balanced dataset."""

    NUM_CLASSES: int = 47
    r"""The number of classes in EMNIST Balanced dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of EMNIST Balanced images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of EMNIST Balanced images (except channel)."""

    MEAN: tuple[float] = (0.1754,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3336,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # digit 0
        1: 1,  # digit 1
        2: 2,  # digit 2
        3: 3,  # digit 3
        4: 4,  # digit 4
        5: 5,  # digit 5
        6: 6,  # digit 6
        7: 7,  # digit 7
        8: 8,  # digit 8
        9: 9,  # digit 9
        10: 10,  # letter A/a
        11: 11,  # letter B/b
        12: 12,  # letter C/c
        13: 13,  # letter D/d
        14: 14,  # letter E/e
        15: 15,  # letter F/f
        16: 16,  # letter G/g
        17: 17,  # letter H/h
        18: 18,  # letter I/i
        19: 19,  # letter J/j
        20: 20,  # letter K/k
        21: 21,  # letter L/l
        22: 22,  # letter M/m
        23: 23,  # letter N/n
        24: 24,  # letter O/o
        25: 25,  # letter P/p
        26: 26,  # letter Q/q
        27: 27,  # letter R/r
        28: 28,  # letter S/s
        29: 29,  # letter T/t
        30: 30,  # letter U/u
        31: 31,  # letter V/v
        32: 32,  # letter W/w
        33: 33,  # letter X/x
        34: 34,  # letter Y/y
        35: 35,  # letter Z/z
        36: 36,  # letter a
        37: 37,  # letter b
        38: 38,  # letter d
        39: 39,  # letter e
        40: 40,  # letter f
        41: 41,  # letter g
        42: 42,  # letter h
        43: 43,  # letter n
        44: 44,  # letter q
        45: 45,  # letter r
        46: 46,  # letter t
    }
    r"""The mapping from class index to original label name. They correspond to the digits 0-9 and letters A-Z, a-z. The letters a, b, d, e, f, g, h, n, q, r, t are merged with their capital letters."""


class EMNISTLettersConstants(DatasetConstants):
    r"""Constants about EMNIST Letters dataset."""

    NUM_CLASSES: int = 26
    r"""The number of classes in EMNIST Letters dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of EMNIST Letters images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of EMNIST Letters images (except channel)."""

    MEAN: tuple[float] = (0.1722,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3310,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 1,  # letter A
        1: 2,  # letter B
        2: 3,  # letter C
        3: 4,  # letter D
        4: 5,  # letter E
        5: 6,  # letter F
        6: 7,  # letter G
        7: 8,  # letter H
        8: 9,  # letter I
        9: 10,  # letter J
        10: 11,  # letter K
        11: 12,  # letter L
        12: 13,  # letter M
        13: 14,  # letter N
        14: 15,  # letter O
        15: 16,  # letter P
        16: 17,  # letter Q
        17: 18,  # letter R
        18: 19,  # letter S
        19: 20,  # letter T
        20: 21,  # letter U
        21: 22,  # letter V
        22: 23,  # letter W
        23: 24,  # letter X
        24: 25,  # letter Y
        25: 26,  # letter Z
    }
    r"""The mapping from class index to original label name. They correspond to the letters A-Z."""


class EMNISTDigitsConstants(DatasetConstants):
    r"""Constants about EMNIST Digits dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in EMNIST Digits dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of EMNIST Digits images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of EMNIST Digits images (except channel)."""

    MEAN: tuple[float] = (0.1736,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3317,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # digit 0
        1: 1,  # digit 1
        2: 2,  # digit 2
        3: 3,  # digit 3
        4: 4,  # digit 4
        5: 5,  # digit 5
        6: 6,  # digit 6
        7: 7,  # digit 7
        8: 8,  # digit 8
        9: 9,  # digit 9
    }
    r"""The mapping from class index to original label name. They correspond to the digits 0-9."""


class EuroSATConstants(DatasetConstants):
    r"""Constants about EuroSAT dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in EuroSAT dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of EuroSAT images."""

    IMG_SIZE: torch.Size = torch.Size([64, 64])
    r"""The size of EuroSAT images (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Country211 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Country211 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # Annual Crop
        1: 1,  # Forest
        2: 2,  # Herbaceous Vegetation
        3: 3,  # Highway
        4: 4,  # Industrial
        5: 5,  # Lake
        6: 6,  # Pasture
        7: 7,  # Permanent Crop
        8: 8,  # Residential
        9: 9,  # River
    }
    r"""The mapping from class index to original label name. They correspond to the classes in EuroSAT dataset."""


class FaceScrub10Constants(DatasetConstants):
    r"""Constants about FaceScrub-10 dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in FaceScrub-10 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of FaceScrub-10 images."""

    IMG_SIZE: torch.Size = torch.Size([32, 32])
    r"""The size of FaceScrub-10 images (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. FaceScrub-10 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. FaceScrub-10 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the 10 classes in FaceScrub-10 dataset."""


class FaceScrub20Constants(DatasetConstants):
    r"""Constants about FaceScrub-20 dataset."""

    NUM_CLASSES: int = 20
    r"""The number of classes in FaceScrub-20 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of FaceScrub-20 images."""

    IMG_SIZE: torch.Size = torch.Size([32, 32])
    r"""The size of FaceScrub-20 images (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. FaceScrub-20 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. FaceScrub-20 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the 20 classes in FaceScrub-20 dataset."""


class FaceScrub50Constants(DatasetConstants):
    r"""Constants about FaceScrub-50 dataset."""

    NUM_CLASSES: int = 50
    r"""The number of classes in FaceScrub-50 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of FaceScrub-50 images."""

    IMG_SIZE: torch.Size = torch.Size([32, 32])
    r"""The size of FaceScrub-50 images (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. FaceScrub-50 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. FaceScrub-50 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the 50 classes in FaceScrub-50 dataset."""


class FaceScrub100Constants(DatasetConstants):
    r"""Constants about FaceScrub-50 dataset."""

    NUM_CLASSES: int = 100
    r"""The number of classes in FaceScrub-100 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of FaceScrub-100 images."""

    IMG_SIZE: torch.Size = torch.Size([32, 32])
    r"""The size of FaceScrub-100 images (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. FaceScrub-100 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. FaceScrub-100 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the 100 classes in FaceScrub-100 dataset."""


class FashionMNISTConstants(DatasetConstants):
    r"""Constants about Fashion-MNIST dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in Fashion-MNIST dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of Fashion-MNIST images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of Fashion-MNIST images (except channel)."""

    MEAN: tuple[float] = (0.2860,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3530,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # T-shirt/top
        1: 1,  # trouser
        2: 2,  # pullover
        3: 3,  # dress
        4: 4,  # coat
        5: 5,  # sandal
        6: 6,  # shirt
        7: 7,  # sneaker
        8: 8,  # bag
        9: 9,  # ankle boot
    }
    r"""The mapping from class index to original label name. They correspond to the classes in Fashion-MNIST dataset."""


class FER2013Constants(DatasetConstants):
    r"""Constants about FER2013 dataset."""

    NUM_CLASSES: int = 7
    r"""The number of classes in FER2013 dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of FER2013 images."""

    IMG_SIZE: torch.Size = torch.Size([48, 48])
    r"""The size of FER2013 images (except channel)."""

    MEAN: tuple[float] = (0.4850,)
    r"""The mean values of each channel. FER2013 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290,)
    r"""The standard deviation values of each channel. FER2013 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # Anger
        1: 1,  # Disgust
        2: 2,  # Fear
        3: 3,  # Happiness
        4: 4,  # Sadness
        5: 5,  # Surprise
        6: 6,  # Neutral
    }
    r"""The mapping from class index to original label name. They correspond to the expression classes in FER2013 dataset."""


class FGVCAircraftVariantConstants(DatasetConstants):
    r"""Constants about FGVC-Aircraft dataset annotated by variant."""

    NUM_CLASSES: int = 100
    r"""The number of classes in FGVC-Aircraft dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of FGVC-Aircraft images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. FGVC-Aircraft does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. FGVC-Aircraft does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the 100 aircraft variants in FGVC-Aircraft dataset."""


class FGVCAircraftFamilyConstants(DatasetConstants):
    r"""Constants about FGVC-Aircraft dataset annotated by family."""

    NUM_CLASSES: int = 70
    r"""The number of classes in FGVC-Aircraft dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of FGVC-Aircraft images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. FGVC-Aircraft does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. FGVC-Aircraft does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the 70 aircraft families in FGVC-Aircraft dataset."""


class FGVCAircraftManufacturerConstants(DatasetConstants):
    r"""Constants about FGVC-Aircraft dataset annotated by manufacturer."""

    NUM_CLASSES: int = 41
    r"""The number of classes in FGVC-Aircraft dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of FGVC-Aircraft images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. FGVC-Aircraft does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. FGVC-Aircraft does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the 41 aircraft manufacturers in FGVC-Aircraft dataset."""


class Flowers102Constants(DatasetConstants):
    r"""Constants about Flowers102 dataset."""

    NUM_CLASSES: int = 102
    r"""The number of classes in Flowers102 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Flowers102 images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Flowers102 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Flowers102 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # Anemone
        1: 1,  # Apple Blossom
        2: 2,  # Aquilegia
        3: 3,  # Aubrieta
        4: 4,  # Azalea
        5: 5,  # Balsam
        6: 6,  # Bellflower
        7: 7,  # Black-eyed Susan
        8: 8,  # Bluebell
        9: 9,  # Buttercup
        10: 10,  # Calochortus
        11: 11,  # Canna Lily
        12: 12,  # Cineraria
        13: 13,  # Coltsfoot
        14: 14,  # Corn Poppy
        15: 15,  # Cowslip
        16: 16,  # Crocus
        17: 17,  # Daffodil
        18: 18,  # Dahlia
        19: 19,  # Fuchsia
        20: 20,  # Gardenia
        21: 21,  # Geranium
        22: 22,  # Gerbera
        23: 23,  # Gloxinia
        24: 24,  # Hibiscus
        25: 25,  # Iris
        26: 26,  # Jasmine
        27: 27,  # Kalanchoe
        28: 28,  # Lantana
        29: 29,  # Larkspur
        30: 30,  # Lily
        31: 31,  # Lupine
        32: 32,  # Magnolia
        33: 33,  # Marigold
        34: 34,  # Morning Glory
        35: 35,  # Nasturtium
        36: 36,  # Nicotiana
        37: 37,  # Pansy
        38: 38,  # Peony
        39: 39,  # Petunia
        40: 40,  # Phlox
        41: 41,  # Poppy
        42: 42,  # Primula
        43: 43,  # Red Hot Poker
        44: 44,  # Rhododendron
        45: 45,  # Roses
        46: 46,  # Snowdrop
        47: 47,  # Sunflower
        48: 48,  # Sweet Pea
        49: 49,  # Tiger Lily
        50: 50,  # Tulip
        51: 51,  # Wallflower
        52: 52,  # Water Lily
        53: 53,  # Wisteria
        54: 54,  # Yarrow
        55: 55,  # Zinnia
        56: 56,  # Cherry Blossom
        57: 57,  # Dandelion
        58: 58,  # Bluebell (Hyacinthoides non-scripta)
        59: 59,  # Morning Glory (Ipomoea purpurea)
        60: 60,  # Jasmine (Jasminum spp.)
        61: 61,  # Goldenrods
        62: 62,  # Common Lilac
        63: 63,  # Wisteria
        64: 64,  # Japanese Iris
        65: 65,  # Siberian Iris
        66: 66,  # California Poppy
        67: 67,  # Clematis
        68: 68,  # Lobelia
        69: 69,  # Lavender
        70: 70,  # Orange Marigold
        71: 71,  # Hydrangea
        72: 72,  # Mimosa
        73: 73,  # Black-eyed Susan (Rudbeckia hirta)
        74: 74,  # Bellis perennis
        75: 75,  # Daffodil (Narcissus)
        76: 76,  # Ageratum
        77: 77,  # Lotus
        78: 78,  # Hellebore
        79: 79,  # Dogwood
        80: 80,  # Trumpet Creeper
        81: 81,  # Indian Paintbrush
        82: 82,  # Chrysanthemums
        83: 83,  # Pansy
        84: 84,  # Petunias
        85: 85,  # Sweet William
        86: 86,  # Wormwood
        87: 87,  # Forget-me-not
        88: 88,  # Monkey Orchid
        89: 89,  # Gladiolus
        90: 90,  # Bleeding Heart
        91: 91,  # Periwinkle
        92: 92,  # Cineraria
        93: 93,  # Primrose
        94: 94,  # Bachelor Button
        95: 95,  # Poinsettia
        96: 96,  # Chicory
        97: 97,  # Cucumber
        98: 98,  # Borage
        99: 99,  # Papaver
        100: 100,  # Lobelia
        101: 101,  # Butterfly Weed
    }
    r"""The mapping from class index to original label name. They correspond to the classes in Flowers102 dataset."""


class Food101Constants(DatasetConstants):
    r"""Constants about Food-101 dataset."""

    NUM_CLASSES: int = 101
    r"""The number of classes in Food-101 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Food-101 images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Food-101 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Food-101 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # apple pie
        1: 1,  # baby back ribs
        2: 2,  # baklava
        3: 3,  # banana bread
        4: 4,  # beef carpaccio
        5: 5,  # beef tartare
        6: 6,  # belgian waffle
        7: 7,  # bibimbap
        8: 8,  # bread pudding
        9: 9,  # breakfast burrito
        10: 10,  # bruschetta
        11: 11,  # caesar salad
        12: 12,  # cannoli
        13: 13,  # caprese salad
        14: 14,  # carrot cake
        15: 15,  # cauliflower pizza crust
        16: 16,  # cheese platter
        17: 17,  # cheesecake
        18: 18,  # cherry pie
        19: 19,  # chicken alfredo
        20: 20,  # chicken casserole
        21: 21,  # chicken creole
        22: 22,  # chicken kebab
        23: 23,  # chicken marsala
        24: 24,  # chicken parmesan
        25: 25,  # chicken pot pie
        26: 26,  # chicken quesadilla
        27: 27,  # chicken soup
        28: 28,  # chicken tikka masala
        29: 29,  # chocolate cake
        30: 30,  # chocolate mousse
        31: 31,  # churros
        32: 32,  # clam chowder
        33: 33,  # club sandwich
        34: 34,  # crab cakes
        35: 35,  # crispy fried chicken
        36: 36,  # croque madame
        37: 37,  # cupcakes
        38: 38,  # deviled eggs
        39: 39,  # donuts
        40: 40,  # dumplings
        41: 41,  # edamame
        42: 42,  # eggs benedict
        43: 43,  # eggs florentine
        44: 44,  # empanadas
        45: 45,  # falafel
        46: 46,  # filet mignon
        47: 47,  # fish and chips
        48: 48,  # foie gras
        49: 49,  # french fries
        50: 50,  # french onion soup
        51: 51,  # garlic bread
        52: 52,  # gnocchi
        53: 53,  # grilled cheese sandwich
        54: 54,  # guacamole
        55: 55,  # hamburger
        56: 56,  # hot and sour soup
        57: 57,  # hot dog
        58: 58,  # hummus
        59: 59,  # ice cream
        60: 60,  # irish coffee
        61: 61,  # lasagna
        62: 62,  # lobster bisque
        63: 63,  # lobster roll
        64: 64,  # macaroni and cheese
        65: 65,  # maki sushi
        66: 66,  # meatballs
        67: 67,  # mexican rice
        68: 68,  # moussaka
        69: 69,  # nachos
        70: 70,  # omelette
        71: 71,  # onion rings
        72: 72,  # oreos
        73: 73,  # pad thai
        74: 74,  # pasta
        75: 75,  # pasta primavera
        76: 76,  # pasta salad
        77: 77,  # patatas bravas
        78: 78,  # pecan pie
        79: 79,  # peking duck
        80: 80,  # pho
        81: 81,  # pizza
        82: 82,  # pork schnitzel
        83: 83,  # pork tenderloin
        84: 84,  # pot roast
        85: 85,  # poutine
        86: 86,  # pulled pork sandwich
        87: 87,  # ramen
        88: 88,  # risotto
        89: 89,  # samosa
        90: 90,  # sauerbraten
        91: 91,  # seaweed salad
        92: 92,  # shrimp and grits
        93: 93,  # sloppy joes
        94: 94,  # sushi
        95: 95,  # tacos
        96: 96,  # takoyaki
        97: 97,  # tiramisu
        98: 98,  # tom yum
        99: 99,  # tuna tartare
        100: 100,  # waffles
    }
    r"""The mapping from class index to original label name. They correspond to the classes in Food-101 dataset."""


class GTSRBConstants(DatasetConstants):
    r"""Constants about GTSRB dataset."""

    NUM_CLASSES: int = 43
    r"""The number of classes in GTSRB dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of GTSRB images."""

    IMG_SIZE: torch.Size = torch.Size([32, 32])
    r"""The size of GTSRB images (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. GTSRB does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. GTSRB does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # Speed limit (20km/h)
        1: 1,  # Speed limit (30km/h)
        2: 2,  # Speed limit (50km/h)
        3: 3,  # Speed limit (60km/h)
        4: 4,  # Speed limit (70km/h)
        5: 5,  # Speed limit (80km/h)
        6: 6,  # End of speed limit (80km/h)
        7: 7,  # Speed limit (100km/h)
        8: 8,  # Speed limit (120km/h)
        9: 9,  # No passing
        10: 10,  # No passing for vehicles over 3.5 metric tons
        11: 11,  # Right-of-way at the next intersection
        12: 12,  # Priority road
        13: 13,  # Yield
        14: 14,  # Stop
        15: 15,  # No vehicles
        16: 16,  # Vehicles over 3.5 metric tons prohibited
        17: 17,  # No entry
        18: 18,  # General caution
        19: 19,  # Dangerous curve to the left
        20: 20,  # Dangerous curve to the right
        21: 21,  # Double curve
        22: 22,  # Bumpy road
        23: 23,  # Slippery road
        24: 24,  # Road narrows on the right
        25: 25,  # Road work
        26: 26,  # Traffic signals
        27: 27,  # Pedestrians
        28: 28,  # Children crossing
        29: 29,  # Bicycles crossing
        30: 30,  # Beware of ice/snow
        31: 31,  # Wild animals crossing
        32: 32,  # End of all speed and passing limits
        33: 33,  # Turn right ahead
        34: 34,  # Turn left ahead
        35: 35,  # Ahead only
        36: 36,  # Go straight or right
        37: 37,  # Go straight or left
        38: 38,  # Keep right
        39: 39,  # Keep left
        40: 40,  # Roundabout mandatory
        41: 41,  # End of no passing
        42: 42,  # End of no passing by vehicles over 3.5 metric tons
    }
    r"""The mapping from class index to original label name. They correspond to the classes in GTSRB dataset."""


class ImagenetteConstants(DatasetConstants):
    r"""Constants about Imagenette dataset of full size."""

    NUM_CLASSES: int = 10
    r"""The number of classes in Imagenette dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Imagenette images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Because Imagenette is a subset of ImageNet, we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Because Imagenette is a subset of ImageNet, we use the ImageNet mean values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # tench
        1: 1,  # English springer
        2: 2,  # cassette player
        3: 3,  # chain saw
        4: 4,  # church
        5: 5,  # French horn
        6: 6,  # garbage truck
        7: 7,  # gas pump
        8: 8,  # golf ball
        9: 9,  # parachute
    }
    r"""The mapping from class index to original label name. They correspond to the classes in Imagenette dataset."""


class KannadaMNISTConstants(DatasetConstants):
    r"""Constants about Kannada-MNIST dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in Kannada-MNIST dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of Kannada-MNIST images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of Kannada-MNIST images (except channel)."""

    MEAN: tuple[float] = (0.3337,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2760,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # Kannada digit ೦ (0)
        1: 1,  # Kannada digit ೧ (1)
        2: 2,  # Kannada digit ೨ (2)
        3: 3,  # Kannada digit ೩ (3)
        4: 4,  # Kannada digit ೪ (4)
        5: 5,  # Kannada digit ೫ (5)
        6: 6,  # Kannada digit ೬ (6)
        7: 7,  # Kannada digit ೭ (7)
        8: 8,  # Kannada digit ೮ (8)
        9: 9,  # Kannada digit ೯ (9)
    }
    r"""The mapping from class index to original label name. They correspond to the Kannada digits 0-9."""


class KMNISTConstants(DatasetConstants):
    r"""Constants about Kuzushiji-MNIST dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in Kuzushiji-MNIST dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of Kuzushiji-MNIST images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of Kuzushiji-MNIST images (except channel)."""

    MEAN: tuple[float] = (0.1904,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3475,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # Kuzushiji character お
        1: 1,  # Kuzushiji character き
        2: 2,  # Kuzushiji character す
        3: 3,  # Kuzushiji character つ
        4: 4,  # Kuzushiji character な
        5: 5,  # Kuzushiji character は
        6: 6,  # Kuzushiji character ま
        7: 7,  # Kuzushiji character や
        8: 8,  # Kuzushiji character れ
        9: 9,  # Kuzushiji character を
    }
    r"""The mapping from class index to original label name. They correspond to the classes in Kuzushiji-MNIST dataset."""


class Linnaeus5_32Constants(DatasetConstants):
    r"""Constants about Linnaeus 5 dataset (32x32)."""

    NUM_CLASSES: int = 5
    r"""The number of classes in Linnaeus 5 dataset (32x32)."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Linnaeus 5 images."""

    IMG_SIZE: torch.Size = torch.Size([32, 32])
    r"""The size of Linnaeus 5 images (32x32) (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Linnaeus 5 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Linnaeus 5 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # berry
        1: 1,  # birds
        2: 2,  # dog
        3: 3,  # flower
        4: 4,  # other
    }
    r"""The mapping from class index to original label name. They correspond to the classes in Linnaeus 5 dataset."""


class Linnaeus5_64Constants(DatasetConstants):
    r"""Constants about Linnaeus 5 dataset (64x64)."""

    NUM_CLASSES: int = 5
    r"""The number of classes in Linnaeus 5 dataset (64x64)."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Linnaeus 5 images."""

    IMG_SIZE: torch.Size = torch.Size([64, 64])
    r"""The size of Linnaeus 5 images (64x64) (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Linnaeus 5 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Linnaeus 5 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # berry
        1: 1,  # birds
        2: 2,  # dog
        3: 3,  # flower
        4: 4,  # other
    }
    r"""The mapping from class index to original label name. They correspond to the classes in Linnaeus 5 dataset."""


class Linnaeus5_128Constants(DatasetConstants):
    r"""Constants about Linnaeus 5 dataset (128x128)."""

    NUM_CLASSES: int = 5
    r"""The number of classes in Linnaeus 5 dataset (128x128)."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Linnaeus 5 images."""

    IMG_SIZE: torch.Size = torch.Size([128, 128])
    r"""The size of Linnaeus 5 images (128x128) (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Linnaeus 5 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Linnaeus 5 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # berry
        1: 1,  # birds
        2: 2,  # dog
        3: 3,  # flower
        4: 4,  # other
    }
    r"""The mapping from class index to original label name. They correspond to the classes in Linnaeus 5 dataset."""


class Linnaeus5_256Constants(DatasetConstants):
    r"""Constants about Linnaeus 5 dataset (256x256)."""

    NUM_CLASSES: int = 5
    r"""The number of classes in Linnaeus 5 dataset (256x256)."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Linnaeus 5 images."""

    IMG_SIZE: torch.Size = torch.Size([256, 256])
    r"""The size of Linnaeus 5 images (256x256) (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Linnaeus 5 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Linnaeus 5 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # berry
        1: 1,  # birds
        2: 2,  # dog
        3: 3,  # flower
        4: 4,  # other
    }
    r"""The mapping from class index to original label name. They correspond to the classes in Linnaeus 5 dataset."""


class MNISTConstants(DatasetConstants):
    r"""Constants about MNIST dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in MNIST dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of MNIST images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of MNIST images (except channel)."""

    MEAN: tuple[float] = (0.1307,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.3081,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # digit 0
        1: 1,  # digit 1
        2: 2,  # digit 2
        3: 3,  # digit 3
        4: 4,  # digit 4
        5: 5,  # digit 5
        6: 6,  # digit 6
        7: 7,  # digit 7
        8: 8,  # digit 8
        9: 9,  # digit 9
    }
    r"""The mapping from class index to original label name. They correspond to the digits 0-9."""


class NotMNISTConstants(DatasetConstants):
    r"""Constants about NotMNIST dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in NotMNIST dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of NotMNIST images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of NotMNIST images (except channel). """

    MEAN: tuple[float] = (0.4254,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.4501,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 1,  # letter A
        1: 2,  # letter B
        2: 3,  # letter C
        3: 4,  # letter D
        4: 5,  # letter E
        5: 6,  # letter F
        6: 7,  # letter G
        7: 8,  # letter H
        8: 9,  # letter I
        9: 10,  # letter J
    }
    r"""The mapping from class index to original label name. They correspond to the letters A-J."""


class OxfordIIITPet37Constants(DatasetConstants):
    r"""Constants about Oxford Pets dataset with 37 breed classes."""

    NUM_CLASSES: int = 37
    r"""The number of breed classes in Oxford Pets dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Oxford Pets images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Oxford Pets does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Oxford Pets does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # Abyssinian
        1: 1,  # Basset Hound
        2: 2,  # Beagle
        3: 3,  # Bengal
        4: 4,  # Birman
        5: 5,  # Bombay
        6: 6,  # British Shorthair
        7: 7,  # Chihuahua
        8: 8,  # Egyptian Mau
        9: 9,  # English Cocker Spaniel
        10: 10,  # English Setter
        11: 11,  # German Shepherd
        12: 12,  # Golden Retriever
        13: 13,  # Great Dane
        14: 14,  # Havanese
        15: 15,  # Indian Spitz
        16: 16,  # Japanese Chin
        17: 17,  # Keeshond
        18: 18,  # King Charles Spaniel
        19: 19,  # Labrador Retriever
        20: 20,  # Maine Coon
        21: 21,  # Miniature Pinscher
        22: 22,  # Munchkin
        23: 23,  # Norwegian Forest Cat
        24: 24,  # Persian
        25: 25,  # Pomeranian
        26: 26,  # Ragdoll
        27: 27,  # Russian Blue
        28: 28,  # Saint Bernard
        29: 29,  # Samoyed
        30: 30,  # Schnauzer
        31: 31,  # Scottish Terrier
        32: 32,  # Shih Tzu
        33: 33,  # Siamese
        34: 34,  # Siberian Husky
        35: 35,  # Welsh Corgi
        36: 36,  # Yorkshire Terrier
    }
    r"""The mapping from class index to original label name. They correspond to the breed classes in Oxford Pets dataset."""


class OxfordIIITPet2Constants(DatasetConstants):
    r"""Constants about Oxford Pets dataset with binary classes: cat, dog."""

    NUM_CLASSES: int = 2
    r"""The number of breed classes in Oxford Pets dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Oxford Pets images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Oxford Pets does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Oxford Pets does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # cat
        1: 1,  # dog
    }
    r"""The mapping from class index to original label name. Only two classes are used in here: cat and dog."""


class PCAMConstants(DatasetConstants):
    r"""Constants about PCAM dataset."""

    NUM_CLASSES: int = 2
    r"""The number of classes in PCAM dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of PCAM images."""

    IMG_SIZE: torch.Size = torch.Size([96, 96])
    r"""The size of PCAM images (except channel)."""

    MEAN: tuple[float] = (0.7009, 0.5469, 0.6960)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.1645, 0.2044, 0.1616)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # metastatic tissue not present
        1: 1,  # metastatic tissue present
    }
    r"""The mapping from class index to original label name. They correspond to the binary classes in PCAM dataset."""


class RenderedSST2Constants(DatasetConstants):
    r"""Constants about Rendered SST2 dataset."""

    NUM_CLASSES: int = 2
    r"""The number of classes in Rendered SST2 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Rendered SST2 images."""

    IMG_SIZE: torch.Size = torch.Size([448, 448])
    r"""The size of PCAM images (except channel)."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Rendered SST2 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Rendered SST2 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # negative
        1: 1,  # positive
    }
    r"""The mapping from class index to original label name. They correspond to the binary classes in Rendered SST2 dataset."""


class SEMEIONConstants(DatasetConstants):
    r"""Constants about SEMEION dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in SEMEION dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of SEMEION images."""

    IMG_SIZE: torch.Size = torch.Size([16, 16])
    r"""The size of SEMEION images (except channel)."""

    MEAN: tuple[float] = (0.0838,)
    r"""The mean values of each channel."""

    STD: tuple[float] = (0.277,)
    r"""The standard deviation values of each channel."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # digit 0
        1: 1,  # digit 1
        2: 2,  # digit 2
        3: 3,  # digit 3
        4: 4,  # digit 4
        5: 5,  # digit 5
        6: 6,  # digit 6
        7: 7,  # digit 7
        8: 8,  # digit 8
        9: 9,  # digit 9
    }
    r"""The mapping from class index to original label name. They correspond to the digits 0-9."""


class SignLanguageMNISTConstants(DatasetConstants):
    r"""Constants about Sign Language MNIST dataset."""

    NUM_CLASSES: int = 24
    r"""The number of classes in Sign Language MNIST dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of Sign Language MNIST images."""

    IMG_SIZE: torch.Size = torch.Size([28, 28])
    r"""The size of Sign Language MNIST images (except channel)."""

    MEAN: tuple[float] = (0.3079,)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2741,)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # sign language A
        1: 1,  # sign language B
        2: 2,  # sign language C
        3: 3,  # sign language D
        4: 4,  # sign language E
        5: 5,  # sign language F
        6: 6,  # sign language G
        7: 7,  # sign language H
        8: 8,  # sign language I
        # J doesn't have static sign language, skipped
        9: 10,  # sign language K
        10: 11,  # sign language L
        11: 12,  # sign language M
        12: 13,  # sign language N
        13: 14,  # sign language O
        14: 15,  # sign language P
        15: 16,  # sign language Q
        16: 17,  # sign language R
        17: 18,  # sign language S
        18: 19,  # sign language T
        19: 20,  # sign language U
        20: 21,  # sign language V
        21: 22,  # sign language W
        22: 23,  # sign language X
        23: 24,  # sign language Y
        # Z doesn't have static sign language, skipped
    }
    r"""The mapping from class index to original label name. They correspond to the letters A-Y, excluding J and Z."""


class StanfordCarsConstants(DatasetConstants):
    r"""Constants about Stanford Cars dataset."""

    NUM_CLASSES: int = 196
    r"""The number of classes in Stanford Cars dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Stanford Cars images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. Stanford Cars does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. Stanford Cars does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the classes in Stanford Cars dataset."""


class SUN397Constants(DatasetConstants):
    r"""Constants about SUN397 dataset."""

    NUM_CLASSES: int = 397
    r"""The number of classes in SUN397 dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of SUN397 images."""

    MEAN: tuple[float] = (0.4850, 0.4560, 0.4060)
    r"""The mean values of each channel. SUN397 does not have official mean values, so we use the ImageNet mean values."""

    STD: tuple[float] = (0.2290, 0.2240, 0.2250)
    r"""The standard deviation values of each channel. SUN397 does not have official std values, so we use the ImageNet std values."""

    CLASS_MAP: dict[int, int] = {i: i for i in range(NUM_CLASSES)}
    r"""The mapping from class index to original label name. They correspond to the classes in SUN397 dataset."""


class SVHNConstants(DatasetConstants):
    r"""Constants about SVHN dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in SVHN dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of SVHN images."""

    IMG_SIZE: torch.Size = torch.Size([32, 32])
    r"""The size of SVHN images (except channel)."""

    MEAN: tuple[float] = (0.4377, 0.4438, 0.4728)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.1980, 0.2010, 0.1970)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # digit 0
        1: 1,  # digit 1
        2: 2,  # digit 2
        3: 3,  # digit 3
        4: 4,  # digit 4
        5: 5,  # digit 5
        6: 6,  # digit 6
        7: 7,  # digit 7
        8: 8,  # digit 8
        9: 9,  # digit 9
    }
    r"""The mapping from class index to original label name. They correspond to the digits 0-9."""


class TinyImageNetConstants(DatasetConstants):
    r"""Constants about TinyImageNet dataset."""

    NUM_CLASSES: int = 200
    r"""The number of classes in Tiny ImageNet dataset."""

    NUM_CHANNELS: int = 3
    r"""The number of channels of Tiny ImageNet images."""

    IMG_SIZE: torch.Size = torch.Size([64, 64])
    r"""The size of Tiny ImageNet images (except channel)."""

    MEAN: tuple[float] = (0.4802, 0.4481, 0.3975)
    r"""The mean values of each channel. """

    STD: tuple[float] = (0.2302, 0.2265, 0.2262)
    r"""The standard deviation values of each channel. """

    CLASS_MAP: dict[int, int] = {
        0: 0,  # n01443537, goldfish
        1: 1,  # n01629819, European fire salamander
        2: 2,  # n01641577, bullfrog
        3: 3,  # n01644900, tailed frog
        4: 4,  # n01698640, American alligator
        5: 5,  # n01742172, boa constrictor
        6: 6,  # n01768244, trilobite
        7: 7,  # n01770393, scorpion
        8: 8,  # n01774384, black and gold garden spider
        9: 9,  # n01774750, tarantula
        10: 10,  # n01784675, centipede
        11: 11,  # n01855672, goose
        12: 12,  # n01882714, koala
        13: 13,  # n01910747, jellyfish
        14: 14,  # n01917289, sea anemone
        15: 15,  # n01944390, snail
        16: 16,  # n01945685, slug
        17: 17,  # n01950731, sea slug
        18: 18,  # n01983481, American lobster
        19: 19,  # n01984695, spiny lobster
        20: 20,  # n02002724, black stork
        21: 21,  # n02056570, king penguin
        22: 22,  # n02058221, albatross
        23: 23,  # n02066245, grey whale
        24: 24,  # n02071294, killer whale
        25: 25,  # n02074367, dugong
        26: 26,  # n02077923, sea lion
        27: 27,  # n02085620, Chihuahua
        28: 28,  # n02094433, Yorkshire terrier
        29: 29,  # n02099601, golden retriever
        30: 30,  # n02099712, Labrador retriever
        31: 31,  # n02106662, German shepherd
        32: 32,  # n02113799, standard poodle
        33: 33,  # n02123045, tabby cat
        34: 34,  # n02123394, Persian cat
        35: 35,  # n02124075, Egyptian cat
        36: 36,  # n02125311, cougar
        37: 37,  # n02129165, lion
        38: 38,  # n02132136, brown bear
        39: 39,  # n02165456, ladybug
        40: 40,  # n02174001, rhinoceros beetle
        41: 41,  # n02177972, weevil
        42: 42,  # n02190166, fly
        43: 43,  # n02206856, bee
        44: 44,  # n02226429, grasshopper
        45: 45,  # n02231487, walking stick
        46: 46,  # n02233338, cockroach
        47: 47,  # n02236044, mantis
        48: 48,  # n02268443, dragonfly
        49: 49,  # n02279972, damselfly
        50: 50,  # n02321529, sea cucumber
        51: 51,  # n02364673, guinea pig
        52: 52,  # n02395406, hog
        53: 53,  # n02403003, ox
        54: 54,  # n02410509, bison
        55: 55,  # n02415577, bighorn
        56: 56,  # n02423022, gazelle
        57: 57,  # n02437312, Arabian camel
        58: 58,  # n02480495, orangutan
        59: 59,  # n02481823, chimpanzee
        60: 60,  # n02486410, baboon
        61: 61,  # n02504458, African elephant
        62: 62,  # n02509815, Indian elephant
        63: 63,  # n02655020, airliner
        64: 64,  # n02692877, airship
        65: 65,  # n02730930, apron
        66: 66,  # n02769748, backpack
        67: 67,  # n02788148, bannister
        68: 68,  # n02795169, barrel
        69: 69,  # n02802426, basketball
        70: 70,  # n02808440, bathtub
        71: 71,  # n02814533, beach wagon
        72: 72,  # n02814860, beacon
        73: 73,  # n02815834, beaker
        74: 74,  # n02823428, beer bottle
        75: 75,  # n02837789, bikini
        76: 76,  # n02841315, binoculars
        77: 77,  # n02843684, birdhouse
        78: 78,  # n02883205, bow tie
        79: 79,  # n02892201, brass
        80: 80,  # n02906734, broom
        81: 81,  # n02909870, bucket
        82: 82,  # n02917067, bullet train
        83: 83,  # n02927161, butcher shop
        84: 84,  # n02948072, candle
        85: 85,  # n02950826, cannon
        86: 86,  # n02963159, cardigan
        87: 87,  # n02977058, cash machine
        88: 88,  # n02978881, cassette
        89: 89,  # n02979186, cassette player
        90: 90,  # n02988304, CD player
        91: 91,  # n03014705, chest
        92: 92,  # n03026506, Christmas stocking
        93: 93,  # n03042490, cliff dwelling
        94: 94,  # n03085013, computer keyboard
        95: 95,  # n03089624, confectionery
        96: 96,  # n03100240, convertible
        97: 97,  # n03126707, crane
        98: 98,  # n03160309, dam
        99: 99,  # n03179701, desk
        100: 100,  # n03201208, dining table
        101: 101,  # n03255030, dumbbell
        102: 102,  # n03355925, flagpole
        103: 103,  # n03388043, fountain
        104: 104,  # n03393912, freight car
        105: 105,  # n03400231, frying pan
        106: 106,  # n03404251, fur coat
        107: 107,  # n03417042, garbage truck
        108: 108,  # n03424325, gas pump
        109: 109,  # n03444034, go-kart
        110: 110,  # n03445777, golf ball
        111: 111,  # n03452741, grand piano
        112: 112,  # n03457902, greenhouse
        113: 113,  # n03461385, grocery store
        114: 114,  # n03467068, guillotine
        115: 115,  # n03529860, home theater
        116: 116,  # n03544143, hourglass
        117: 117,  # n03584254, iPod
        118: 118,  # n03584829, iron
        119: 119,  # n03590841, jack-o'-lantern
        120: 120,  # n03594734, jean
        121: 121,  # n03594945, jeep
        122: 122,  # n03617480, kimono
        123: 123,  # n03637318, lampshade
        124: 124,  # n03649909, lawn mower
        125: 125,  # n03657121, lens cap
        126: 126,  # n03662601, lifeboat
        127: 127,  # n03670208, limousine
        128: 128,  # n03706229, magnetic compass
        129: 129,  # n03733131, maypole
        130: 130,  # n03763968, military uniform
        131: 131,  # n03770439, minivan
        132: 132,  # n03796401, moving van
        133: 133,  # n03814639, neck brace
        134: 134,  # n03837869, obelisk
        135: 135,  # n03868242, oxyacetylene torch
        136: 136,  # n03868863, oxygen mask
        137: 137,  # n03871628, packet
        138: 138,  # n03873416, paddlewheel
        139: 139,  # n03874293, paddle
        140: 140,  # n03877472, pajama
        141: 141,  # n03877845, palace
        142: 142,  # n03891332, parking meter
        143: 143,  # n03895866, passenger car
        144: 144,  # n03899768, patio
        145: 145,  # n03902125, pay-phone
        146: 146,  # n03903868, pedestal
        147: 147,  # n03908618, pencil box
        148: 148,  # n03930313, picket fence
        149: 149,  # n03937543, pill bottle
        150: 150,  # n03970156, plunger
        151: 151,  # n03976467, Polaroid camera
        152: 152,  # n03980874, poncho
        153: 153,  # n03982430, pool table
        154: 154,  # n03983396, pop bottle
        155: 155,  # n03992509, potter's wheel
        156: 156,  # n03995372, power drill
        157: 157,  # n04004767, printer
        158: 158,  # n04019541, puck
        159: 159,  # n04023962, punching bag
        160: 160,  # n04026417, purse
        161: 161,  # n04033901, quill
        162: 162,  # n04033995, quonset
        163: 163,  # n04044716, radio telescope
        164: 164,  # n04069434, reflex camera
        165: 165,  # n04086273, revolver
        166: 166,  # n04118538, rugby ball
        167: 167,  # n04120489, running shoe
        168: 168,  # n04125021, safe
        169: 169,  # n04141327, saddle
        170: 170,  # n04146614, school bus
        171: 171,  # n04149813, scoreboard
        172: 172,  # n04179913, sewing machine
        173: 173,  # n04200800, shoe shop
        174: 174,  # n04201297, shoji
        175: 175,  # n04204238, shopping basket
        176: 176,  # n04209133, shower cap
        177: 177,  # n04209239, shower curtain
        178: 178,  # n04228054, ski
        179: 179,  # n04229816, ski mask
        180: 180,  # n04235860, sleeping bag
        181: 181,  # n04238763, slide rule
        182: 182,  # n04251144, snorkel
        183: 183,  # n04259630, sombrero
        184: 184,  # n04263257, soup bowl
        185: 185,  # n04264628, space bar
        186: 186,  # n04275548, spider web
        187: 187,  # n04285008, sports car
        188: 188,  # n04310018, steam locomotive
        189: 189,  # n04311004, steel drum
        190: 190,  # n04328186, stopwatch
        191: 191,  # n04356056, sunglasses
        192: 192,  # n04357314, sunscreen
        193: 193,  # n04366367, suspension bridge
        194: 194,  # n04376876, syringe
        195: 195,  # n04398044, teapot
        196: 196,  # n04399382, teddy bear
        197: 197,  # n04417672, thatch
        198: 198,  # n04456115, torch
        199: 199,  # n04486054, toyshop
    }
    r"""The mapping from class index to original label name. They correspond to the classes in Tiny ImageNet dataset."""


class USPSConstants(DatasetConstants):
    r"""Constants about USPS dataset."""

    NUM_CLASSES: int = 10
    r"""The number of classes in USPS dataset."""

    NUM_CHANNELS: int = 1
    r"""The number of channels of USPS images."""

    IMG_SIZE: torch.Size = torch.Size([16, 16])
    r"""The size of USPS images (except channel)."""

    CLASS_MAP: dict[int, int] = {
        0: 0,  # digit 0
        1: 1,  # digit 1
        2: 2,  # digit 2
        3: 3,  # digit 3
        4: 4,  # digit 4
        5: 5,  # digit 5
        6: 6,  # digit 6
        7: 7,  # digit 7
        8: 8,  # digit 8
        9: 9,  # digit 9
    }
    r"""The mapping from class index to original label name. They correspond to the digits 0-9."""


DATASET_CONSTANTS_MAPPING: dict[type[Dataset], type[DatasetConstants]] = {
    torchvision.datasets.MNIST: MNISTConstants,
    raw.EMNISTByClass: EMNISTByClassConstants,
    raw.EMNISTByMerge: EMNISTByMergeConstants,
    raw.EMNISTBalanced: EMNISTBalancedConstants,
    raw.EMNISTLetters: EMNISTLettersConstants,
    raw.EMNISTDigits: EMNISTDigitsConstants,
    raw.SignLanguageMNIST: SignLanguageMNISTConstants,
    raw.ArabicHandwrittenDigits: ArabicHandwrittenDigitsConstants,
    raw.KannadaMNIST: KannadaMNISTConstants,
    raw.NotMNIST: NotMNISTConstants,
    raw.NotMNISTFromHAT: NotMNISTConstants,
    torchvision.datasets.Country211: Country211Constants,
    raw.FGVCAircraftVariant: FGVCAircraftVariantConstants,
    raw.FGVCAircraftFamily: FGVCAircraftFamilyConstants,
    raw.FGVCAircraftManufacturer: FGVCAircraftManufacturerConstants,
    raw.FaceScrub10: FaceScrub10Constants,
    raw.FaceScrub20: FaceScrub20Constants,
    raw.FaceScrub50: FaceScrub50Constants,
    raw.FaceScrub100: FaceScrub100Constants,
    raw.FaceScrubFromHAT: FaceScrub100Constants,
    raw.OxfordIIITPet37: OxfordIIITPet37Constants,
    raw.OxfordIIITPet2: OxfordIIITPet2Constants,
    torchvision.datasets.PCAM: PCAMConstants,
    torchvision.datasets.StanfordCars: StanfordCarsConstants,
    torchvision.datasets.RenderedSST2: RenderedSST2Constants,
    torchvision.datasets.SUN397: SUN397Constants,
    torchvision.datasets.USPS: USPSConstants,
    torchvision.datasets.SEMEION: SEMEIONConstants,
    torchvision.datasets.Flowers102: Flowers102Constants,
    torchvision.datasets.Food101: Food101Constants,
    torchvision.datasets.DTD: DTDConstants,
    torchvision.datasets.EuroSAT: EuroSATConstants,
    torchvision.datasets.Caltech101: Caltech101Constants,
    torchvision.datasets.Caltech256: Caltech256Constants,
    torchvision.datasets.FashionMNIST: FashionMNISTConstants,
    torchvision.datasets.KMNIST: KMNISTConstants,
    torchvision.datasets.FER2013: FER2013Constants,
    torchvision.datasets.SVHN: SVHNConstants,
    torchvision.datasets.CIFAR10: CIFAR10Constants,
    torchvision.datasets.CIFAR100: CIFAR100Constants,
    torchvision.datasets.CelebA: CelebAConstants,
    tinyimagenet.TinyImageNet: TinyImageNetConstants,
    torchvision.datasets.Imagenette: ImagenetteConstants,
    raw.CUB2002011: CUB2002011Constants,
    torchvision.datasets.GTSRB: GTSRBConstants,
    raw.TrafficSignsFromHAT: GTSRBConstants,
    raw.Linnaeus5_32: Linnaeus5_32Constants,
    raw.Linnaeus5_64: Linnaeus5_64Constants,
    raw.Linnaeus5_128: Linnaeus5_128Constants,
    raw.Linnaeus5_256: Linnaeus5_256Constants,
}
r"""A dictionary that maps dataset classes to their corresponding constants."""
