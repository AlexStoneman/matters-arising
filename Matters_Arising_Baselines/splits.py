import numpy as np
import random


class LNP_Dataset:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test) -> None:
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


murcko_scaffold_indices = {
    "train" : [0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53, 54, 80, 81, 82, 83, 84, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 150, 151, 152, 153, 154, 180, 181, 182, 183, 184, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 250, 251, 252, 253, 254, 280, 281, 282, 283, 284, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 350, 351, 352, 353, 354, 380, 381, 382, 383, 384, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 450, 451, 452, 453, 454, 480, 481, 482, 483, 484, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 550, 551, 552, 553, 554, 580, 581, 582, 583, 584, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 650, 651, 652, 653, 654, 680, 681, 682, 683, 684, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 750, 751, 752, 753, 754, 780, 781, 782, 783, 784, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 850, 851, 852, 853, 854, 880, 881, 882, 883, 884, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 950, 951, 952, 953, 954, 980, 981, 982, 983, 984, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1050, 1051, 1052, 1053, 1054, 1080, 1081, 1082, 1083, 1084, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1150, 1151, 1152, 1153, 1154, 1180, 1181, 1182, 1183, 1184, 1195, 1196, 1197, 1198, 1199, 15, 16, 17, 18, 19, 60, 61, 62, 63, 64, 75, 76, 77, 78, 79, 90, 91, 92, 93, 94, 115, 116, 117, 118, 119, 160, 161, 162, 163, 164, 175, 176, 177, 178, 179, 190, 191, 192, 193, 194, 215, 216, 217, 218, 219, 260, 261, 262, 263, 264, 275, 276, 277, 278, 279, 290, 291, 292, 293, 294, 315, 316, 317, 318, 319, 360, 361, 362, 363, 364, 375, 376, 377, 378, 379, 390, 391, 392, 393, 394, 415, 416, 417, 418, 419, 460, 461, 462, 463, 464, 475, 476, 477, 478, 479, 490, 491, 492, 493, 494, 515, 516, 517, 518, 519, 560, 561, 562, 563, 564, 575, 576, 577, 578, 579, 590, 591, 592, 593, 594, 615, 616, 617, 618, 619, 660, 661, 662, 663, 664, 675, 676, 677, 678, 679, 690, 691, 692, 693, 694, 715, 716, 717, 718, 719, 760, 761, 762, 763, 764, 775, 776, 777, 778, 779, 790, 791, 792, 793, 794, 815, 816, 817, 818, 819, 860, 861, 862, 863, 864, 875, 876, 877, 878, 879, 890, 891, 892, 893, 894, 915, 916, 917, 918, 919, 960, 961, 962, 963, 964, 975, 976, 977, 978, 979, 990, 991, 992, 993, 994, 1015, 1016, 1017, 1018, 1019, 1060, 1061, 1062, 1063, 1064, 1075, 1076, 1077, 1078, 1079, 1090, 1091, 1092, 1093, 1094, 1115, 1116, 1117, 1118, 1119, 1160, 1161, 1162, 1163, 1164, 1175, 1176, 1177, 1178, 1179, 1190, 1191, 1192, 1193, 1194, 5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109, 140, 141, 142, 143, 144, 185, 186, 187, 188, 189, 205, 206, 207, 208, 209, 240, 241, 242, 243, 244, 285, 286, 287, 288, 289, 305, 306, 307, 308, 309, 340, 341, 342, 343, 344, 385, 386, 387, 388, 389, 405, 406, 407, 408, 409, 440, 441, 442, 443, 444, 485, 486, 487, 488, 489, 505, 506, 507, 508, 509, 540, 541, 542, 543, 544, 585, 586, 587, 588, 589, 605, 606, 607, 608, 609, 640, 641, 642, 643, 644, 685, 686, 687, 688, 689, 705, 706, 707, 708, 709, 740, 741, 742, 743, 744, 785, 786, 787, 788, 789, 805, 806, 807, 808, 809, 840, 841, 842, 843, 844, 885, 886, 887, 888, 889, 905, 906, 907, 908, 909, 940, 941, 942, 943, 944, 985, 986, 987, 988, 989, 1005, 1006, 1007, 1008, 1009, 1040, 1041, 1042, 1043, 1044, 1085, 1086, 1087, 1088, 1089, 1105, 1106, 1107, 1108, 1109, 1140, 1141, 1142, 1143, 1144, 1185, 1186, 1187, 1188, 1189, 10, 11, 12, 13, 14, 110, 111, 112, 113, 114, 210, 211, 212, 213, 214, 310, 311, 312, 313, 314, 410, 411, 412, 413, 414, 510, 511, 512, 513, 514, 610, 611, 612, 613, 614, 710, 711, 712, 713, 714, 810, 811, 812, 813, 814, 910, 911, 912, 913, 914, 1010, 1011, 1012, 1013, 1014, 1110, 1111, 1112, 1113, 1114],
    "val" : [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174],
    "test" : [45, 46, 47, 48, 49, 55, 56, 57, 58, 59, 145, 146, 147, 148, 149, 155, 156, 157, 158, 159, 245, 246, 247, 248, 249, 255, 256, 257, 258, 259, 345, 346, 347, 348, 349, 355, 356, 357, 358, 359, 445, 446, 447, 448, 449, 455, 456, 457, 458, 459, 545, 546, 547, 548, 549, 555, 556, 557, 558, 559, 645, 646, 647, 648, 649, 655, 656, 657, 658, 659, 745, 746, 747, 748, 749, 755, 756, 757, 758, 759, 845, 846, 847, 848, 849, 855, 856, 857, 858, 859, 945, 946, 947, 948, 949, 955, 956, 957, 958, 959, 1045, 1046, 1047, 1048, 1049, 1055, 1056, 1057, 1058, 1059, 1145, 1146, 1147, 1148, 1149, 1155, 1156, 1157, 1158, 1159]
    }


def get_library_scaffold_split(one_hot_mat, seed: int):
    
    random.seed(seed)
    test_amine = random.randint(0,19)
    test_aldehyde = random.randint(20,31)
    val_amine = random.randint(0,19)
    val_aldehyde = random.randint(20,31)
    while val_amine == test_amine:
        val_amine = random.randint(0,19)
    while val_aldehyde == test_aldehyde:
        val_aldehyde = random.randint(20,31)

    # print(test_amine)
    # print(test_aldehyde)
    # print(val_amine)
    # print(val_aldehyde)

    test_amine_mols = np.where(one_hot_mat[:,test_amine] == 1)[0]
    test_aldehyde_mols = np.where(one_hot_mat[:,test_aldehyde] == 1)[0]
    unique_test_mols = np.unique(np.hstack((test_amine_mols, test_aldehyde_mols)))

    val_amine_mols = np.where(one_hot_mat[:,val_amine] == 1)[0]
    val_aldehyde_mols = np.where(one_hot_mat[:,val_aldehyde] == 1)[0]
    unique_val_mols = np.unique(np.hstack((val_amine_mols, val_aldehyde_mols)))
    untested_val_mols = np.setdiff1d(unique_val_mols, unique_test_mols)

    held_out_mols = np.hstack((unique_test_mols, untested_val_mols))
    training_mols = np.setdiff1d(np.arange(0,1200), held_out_mols)

    test_idx = list(unique_test_mols)
    val_idx = list(untested_val_mols)
    train_idx = list(training_mols)
    assert(len(test_idx) == 155)
    assert(len(val_idx) == 145)
    assert(len(train_idx) == 900)

    # print("Training Set:")
    # print(train_idx)    
    # print("Validation Set:")
    # print(val_idx)
    # print("Testing Set:")
    # print(test_idx)

    return train_idx, val_idx, test_idx