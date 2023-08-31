# only standard library

import atexit
import builtins
import os
import pprint
import sys
import termios
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from posix import times_result
from pprint import pformat


def _c_rgb(r: int, g: int, b: int):
    return f"\x1b[38;2;{r};{g};{b}m"


def _c_bg_rgb(r: int, g: int, b: int):
    return f"\x1b[48;2;{r};{g};{b}m"


class Color:
    # https://gitlab.com/dslackw/colored/-/blob/master/colored/library.py

    def rgb(r: int, g: int, b: int):
        return _c_rgb(r, g, b)

    def bg_rgb(r: int, g: int, b: int):
        return _c_bg_rgb(r, g, b)

    reset = "\x1b[m"

    # Bright
    black = "\x1b[90m"
    red = "\x1b[91m"
    green = "\x1b[92m"
    yellow = "\x1b[93m"
    blue = "\x1b[94m"
    magenta = "\x1b[95m"
    cyan = "\x1b[96m"
    white = "\x1b[97m"

    def print(*values, c=green, sep=" ", **kwargs):
        text = ""
        for i, e in enumerate(values):
            text += str(e)
            if i < len(values) - 1:
                text += sep
        builtins.print(c + text + Color.reset, sep=sep, **kwargs)

    def pprint(obj, c=green):
        builtins.print(c + pprint.pformat(obj) + Color.reset)

    # class bold:
    #     black = "\x1b[1;90m"
    #     red = "\x1b[1;91m"
    #     green = "\x1b[1;92m"
    #     yellow = "\x1b[1;93m"
    #     blue = "\x1b[1;94m"
    #     magenta = "\x1b[1;95m"
    #     cyan = "\x1b[1;96m"
    #     white = "\x1b[1;97m"

    # bold = '\033[01m'
    # disable = '\033[02m'
    # underline = '\033[04m'
    # reverse = '\033[07m'
    # strikethrough = '\033[09m'
    # invisible = '\033[08m'

    """
    CSS Colors

    from PIL import ImageColor
    from matplotlib.colors import cnames
    for name, hex in cnames.items():
        c = ImageColor.getcolor(hex, "RGB")
        print(f"{name} = _c_rgb{c}  {hex}")

    https://matplotlib.org/stable/gallery/color/named_colors.html
    """

    class code:
        aliceblue = _c_rgb(240, 248, 255)  # F0F8FF
        antiquewhite = _c_rgb(250, 235, 215)  # FAEBD7
        aqua = _c_rgb(0, 255, 255)  # 00FFFF
        aquamarine = _c_rgb(127, 255, 212)  # 7FFFD4
        azure = _c_rgb(240, 255, 255)  # F0FFFF
        beige = _c_rgb(245, 245, 220)  # F5F5DC
        bisque = _c_rgb(255, 228, 196)  # FFE4C4
        black = _c_rgb(0, 0, 0)  # 000000
        blanchedalmond = _c_rgb(255, 235, 205)  # FFEBCD
        blue = _c_rgb(0, 0, 255)  # 0000FF
        blueviolet = _c_rgb(138, 43, 226)  # 8A2BE2
        brown = _c_rgb(165, 42, 42)  # A52A2A
        burlywood = _c_rgb(222, 184, 135)  # DEB887
        cadetblue = _c_rgb(95, 158, 160)  # 5F9EA0
        chartreuse = _c_rgb(127, 255, 0)  # 7FFF00
        chocolate = _c_rgb(210, 105, 30)  # D2691E
        coral = _c_rgb(255, 127, 80)  # FF7F50
        cornflowerblue = _c_rgb(100, 149, 237)  # 6495ED
        cornsilk = _c_rgb(255, 248, 220)  # FFF8DC
        crimson = _c_rgb(220, 20, 60)  # DC143C
        cyan = _c_rgb(0, 255, 255)  # 00FFFF
        darkblue = _c_rgb(0, 0, 139)  # 00008B
        darkcyan = _c_rgb(0, 139, 139)  # 008B8B
        darkgoldenrod = _c_rgb(184, 134, 11)  # B8860B
        darkgray = _c_rgb(169, 169, 169)  # A9A9A9
        darkgreen = _c_rgb(0, 100, 0)  # 006400
        darkgrey = _c_rgb(169, 169, 169)  # A9A9A9
        darkkhaki = _c_rgb(189, 183, 107)  # BDB76B
        darkmagenta = _c_rgb(139, 0, 139)  # 8B008B
        darkolivegreen = _c_rgb(85, 107, 47)  # 556B2F
        darkorange = _c_rgb(255, 140, 0)  # FF8C00
        darkorchid = _c_rgb(153, 50, 204)  # 9932CC
        darkred = _c_rgb(139, 0, 0)  # 8B0000
        darksalmon = _c_rgb(233, 150, 122)  # E9967A
        darkseagreen = _c_rgb(143, 188, 143)  # 8FBC8F
        darkslateblue = _c_rgb(72, 61, 139)  # 483D8B
        darkslategray = _c_rgb(47, 79, 79)  # 2F4F4F
        darkslategrey = _c_rgb(47, 79, 79)  # 2F4F4F
        darkturquoise = _c_rgb(0, 206, 209)  # 00CED1
        darkviolet = _c_rgb(148, 0, 211)  # 9400D3
        deeppink = _c_rgb(255, 20, 147)  # FF1493
        deepskyblue = _c_rgb(0, 191, 255)  # 00BFFF
        dimgray = _c_rgb(105, 105, 105)  # 696969
        dimgrey = _c_rgb(105, 105, 105)  # 696969
        dodgerblue = _c_rgb(30, 144, 255)  # 1E90FF
        firebrick = _c_rgb(178, 34, 34)  # B22222
        floralwhite = _c_rgb(255, 250, 240)  # FFFAF0
        forestgreen = _c_rgb(34, 139, 34)  # 228B22
        fuchsia = _c_rgb(255, 0, 255)  # FF00FF
        gainsboro = _c_rgb(220, 220, 220)  # DCDCDC
        ghostwhite = _c_rgb(248, 248, 255)  # F8F8FF
        gold = _c_rgb(255, 215, 0)  # FFD700
        goldenrod = _c_rgb(218, 165, 32)  # DAA520
        gray = _c_rgb(128, 128, 128)  # 808080
        green = _c_rgb(0, 128, 0)  # 008000
        greenyellow = _c_rgb(173, 255, 47)  # ADFF2F
        grey = _c_rgb(128, 128, 128)  # 808080
        honeydew = _c_rgb(240, 255, 240)  # F0FFF0
        hotpink = _c_rgb(255, 105, 180)  # FF69B4
        indianred = _c_rgb(205, 92, 92)  # CD5C5C
        indigo = _c_rgb(75, 0, 130)  # 4B0082
        ivory = _c_rgb(255, 255, 240)  # FFFFF0
        khaki = _c_rgb(240, 230, 140)  # F0E68C
        lavender = _c_rgb(230, 230, 250)  # E6E6FA
        lavenderblush = _c_rgb(255, 240, 245)  # FFF0F5
        lawngreen = _c_rgb(124, 252, 0)  # 7CFC00
        lemonchiffon = _c_rgb(255, 250, 205)  # FFFACD
        lightblue = _c_rgb(173, 216, 230)  # ADD8E6
        lightcoral = _c_rgb(240, 128, 128)  # F08080
        lightcyan = _c_rgb(224, 255, 255)  # E0FFFF
        lightgoldenrodyellow = _c_rgb(250, 250, 210)  # FAFAD2
        lightgray = _c_rgb(211, 211, 211)  # D3D3D3
        lightgreen = _c_rgb(144, 238, 144)  # 90EE90
        lightgrey = _c_rgb(211, 211, 211)  # D3D3D3
        lightpink = _c_rgb(255, 182, 193)  # FFB6C1
        lightsalmon = _c_rgb(255, 160, 122)  # FFA07A
        lightseagreen = _c_rgb(32, 178, 170)  # 20B2AA
        lightskyblue = _c_rgb(135, 206, 250)  # 87CEFA
        lightslategray = _c_rgb(119, 136, 153)  # 778899
        lightslategrey = _c_rgb(119, 136, 153)  # 778899
        lightsteelblue = _c_rgb(176, 196, 222)  # B0C4DE
        lightyellow = _c_rgb(255, 255, 224)  # FFFFE0
        lime = _c_rgb(0, 255, 0)  # 00FF00
        limegreen = _c_rgb(50, 205, 50)  # 32CD32
        linen = _c_rgb(250, 240, 230)  # FAF0E6
        magenta = _c_rgb(255, 0, 255)  # FF00FF
        maroon = _c_rgb(128, 0, 0)  # 800000
        mediumaquamarine = _c_rgb(102, 205, 170)  # 66CDAA
        mediumblue = _c_rgb(0, 0, 205)  # 0000CD
        mediumorchid = _c_rgb(186, 85, 211)  # BA55D3
        mediumpurple = _c_rgb(147, 112, 219)  # 9370DB
        mediumseagreen = _c_rgb(60, 179, 113)  # 3CB371
        mediumslateblue = _c_rgb(123, 104, 238)  # 7B68EE
        mediumspringgreen = _c_rgb(0, 250, 154)  # 00FA9A
        mediumturquoise = _c_rgb(72, 209, 204)  # 48D1CC
        mediumvioletred = _c_rgb(199, 21, 133)  # C71585
        midnightblue = _c_rgb(25, 25, 112)  # 191970
        mintcream = _c_rgb(245, 255, 250)  # F5FFFA
        mistyrose = _c_rgb(255, 228, 225)  # FFE4E1
        moccasin = _c_rgb(255, 228, 181)  # FFE4B5
        navajowhite = _c_rgb(255, 222, 173)  # FFDEAD
        navy = _c_rgb(0, 0, 128)  # 000080
        oldlace = _c_rgb(253, 245, 230)  # FDF5E6
        olive = _c_rgb(128, 128, 0)  # 808000
        olivedrab = _c_rgb(107, 142, 35)  # 6B8E23
        orange = _c_rgb(255, 165, 0)  # FFA500
        orangered = _c_rgb(255, 69, 0)  # FF4500
        orchid = _c_rgb(218, 112, 214)  # DA70D6
        palegoldenrod = _c_rgb(238, 232, 170)  # EEE8AA
        palegreen = _c_rgb(152, 251, 152)  # 98FB98
        paleturquoise = _c_rgb(175, 238, 238)  # AFEEEE
        palevioletred = _c_rgb(219, 112, 147)  # DB7093
        papayawhip = _c_rgb(255, 239, 213)  # FFEFD5
        peachpuff = _c_rgb(255, 218, 185)  # FFDAB9
        peru = _c_rgb(205, 133, 63)  # CD853F
        pink = _c_rgb(255, 192, 203)  # FFC0CB
        plum = _c_rgb(221, 160, 221)  # DDA0DD
        powderblue = _c_rgb(176, 224, 230)  # B0E0E6
        purple = _c_rgb(128, 0, 128)  # 800080
        rebeccapurple = _c_rgb(102, 51, 153)  # 663399
        red = _c_rgb(255, 0, 0)  # FF0000
        rosybrown = _c_rgb(188, 143, 143)  # BC8F8F
        royalblue = _c_rgb(65, 105, 225)  # 4169E1
        saddlebrown = _c_rgb(139, 69, 19)  # 8B4513
        salmon = _c_rgb(250, 128, 114)  # FA8072
        sandybrown = _c_rgb(244, 164, 96)  # F4A460
        seagreen = _c_rgb(46, 139, 87)  # 2E8B57
        seashell = _c_rgb(255, 245, 238)  # FFF5EE
        sienna = _c_rgb(160, 82, 45)  # A0522D
        silver = _c_rgb(192, 192, 192)  # C0C0C0
        skyblue = _c_rgb(135, 206, 235)  # 87CEEB
        slateblue = _c_rgb(106, 90, 205)  # 6A5ACD
        slategray = _c_rgb(112, 128, 144)  # 708090
        slategrey = _c_rgb(112, 128, 144)  # 708090
        snow = _c_rgb(255, 250, 250)  # FFFAFA
        springgreen = _c_rgb(0, 255, 127)  # 00FF7F
        steelblue = _c_rgb(70, 130, 180)  # 4682B4
        tan = _c_rgb(210, 180, 140)  # D2B48C
        teal = _c_rgb(0, 128, 128)  # 008080
        thistle = _c_rgb(216, 191, 216)  # D8BFD8
        tomato = _c_rgb(255, 99, 71)  # FF6347
        turquoise = _c_rgb(64, 224, 208)  # 40E0D0
        violet = _c_rgb(238, 130, 238)  # EE82EE
        wheat = _c_rgb(245, 222, 179)  # F5DEB3
        white = _c_rgb(255, 255, 255)  # FFFFFF
        whitesmoke = _c_rgb(245, 245, 245)  # F5F5F5
        yellow = _c_rgb(255, 255, 0)  # FFFF00
        yellowgreen = _c_rgb(154, 205, 50)  # 9ACD32

        class bg:
            aliceblue = _c_bg_rgb(240, 248, 255)  # F0F8FF
            antiquewhite = _c_bg_rgb(250, 235, 215)  # FAEBD7
            aqua = _c_bg_rgb(0, 255, 255)  # 00FFFF
            aquamarine = _c_bg_rgb(127, 255, 212)  # 7FFFD4
            azure = _c_bg_rgb(240, 255, 255)  # F0FFFF
            beige = _c_bg_rgb(245, 245, 220)  # F5F5DC
            bisque = _c_bg_rgb(255, 228, 196)  # FFE4C4
            black = _c_bg_rgb(0, 0, 0)  # 000000
            blanchedalmond = _c_bg_rgb(255, 235, 205)  # FFEBCD
            blue = _c_bg_rgb(0, 0, 255)  # 0000FF
            blueviolet = _c_bg_rgb(138, 43, 226)  # 8A2BE2
            brown = _c_bg_rgb(165, 42, 42)  # A52A2A
            burlywood = _c_bg_rgb(222, 184, 135)  # DEB887
            cadetblue = _c_bg_rgb(95, 158, 160)  # 5F9EA0
            chartreuse = _c_bg_rgb(127, 255, 0)  # 7FFF00
            chocolate = _c_bg_rgb(210, 105, 30)  # D2691E
            coral = _c_bg_rgb(255, 127, 80)  # FF7F50
            cornflowerblue = _c_bg_rgb(100, 149, 237)  # 6495ED
            cornsilk = _c_bg_rgb(255, 248, 220)  # FFF8DC
            crimson = _c_bg_rgb(220, 20, 60)  # DC143C
            cyan = _c_bg_rgb(0, 255, 255)  # 00FFFF
            darkblue = _c_bg_rgb(0, 0, 139)  # 00008B
            darkcyan = _c_bg_rgb(0, 139, 139)  # 008B8B
            darkgoldenrod = _c_bg_rgb(184, 134, 11)  # B8860B
            darkgray = _c_bg_rgb(169, 169, 169)  # A9A9A9
            darkgreen = _c_bg_rgb(0, 100, 0)  # 006400
            darkgrey = _c_bg_rgb(169, 169, 169)  # A9A9A9
            darkkhaki = _c_bg_rgb(189, 183, 107)  # BDB76B
            darkmagenta = _c_bg_rgb(139, 0, 139)  # 8B008B
            darkolivegreen = _c_bg_rgb(85, 107, 47)  # 556B2F
            darkorange = _c_bg_rgb(255, 140, 0)  # FF8C00
            darkorchid = _c_bg_rgb(153, 50, 204)  # 9932CC
            darkred = _c_bg_rgb(139, 0, 0)  # 8B0000
            darksalmon = _c_bg_rgb(233, 150, 122)  # E9967A
            darkseagreen = _c_bg_rgb(143, 188, 143)  # 8FBC8F
            darkslateblue = _c_bg_rgb(72, 61, 139)  # 483D8B
            darkslategray = _c_bg_rgb(47, 79, 79)  # 2F4F4F
            darkslategrey = _c_bg_rgb(47, 79, 79)  # 2F4F4F
            darkturquoise = _c_bg_rgb(0, 206, 209)  # 00CED1
            darkviolet = _c_bg_rgb(148, 0, 211)  # 9400D3
            deeppink = _c_bg_rgb(255, 20, 147)  # FF1493
            deepskyblue = _c_bg_rgb(0, 191, 255)  # 00BFFF
            dimgray = _c_bg_rgb(105, 105, 105)  # 696969
            dimgrey = _c_bg_rgb(105, 105, 105)  # 696969
            dodgerblue = _c_bg_rgb(30, 144, 255)  # 1E90FF
            firebrick = _c_bg_rgb(178, 34, 34)  # B22222
            floralwhite = _c_bg_rgb(255, 250, 240)  # FFFAF0
            forestgreen = _c_bg_rgb(34, 139, 34)  # 228B22
            fuchsia = _c_bg_rgb(255, 0, 255)  # FF00FF
            gainsboro = _c_bg_rgb(220, 220, 220)  # DCDCDC
            ghostwhite = _c_bg_rgb(248, 248, 255)  # F8F8FF
            gold = _c_bg_rgb(255, 215, 0)  # FFD700
            goldenrod = _c_bg_rgb(218, 165, 32)  # DAA520
            gray = _c_bg_rgb(128, 128, 128)  # 808080
            green = _c_bg_rgb(0, 128, 0)  # 008000
            greenyellow = _c_bg_rgb(173, 255, 47)  # ADFF2F
            grey = _c_bg_rgb(128, 128, 128)  # 808080
            honeydew = _c_bg_rgb(240, 255, 240)  # F0FFF0
            hotpink = _c_bg_rgb(255, 105, 180)  # FF69B4
            indianred = _c_bg_rgb(205, 92, 92)  # CD5C5C
            indigo = _c_bg_rgb(75, 0, 130)  # 4B0082
            ivory = _c_bg_rgb(255, 255, 240)  # FFFFF0
            khaki = _c_bg_rgb(240, 230, 140)  # F0E68C
            lavender = _c_bg_rgb(230, 230, 250)  # E6E6FA
            lavenderblush = _c_bg_rgb(255, 240, 245)  # FFF0F5
            lawngreen = _c_bg_rgb(124, 252, 0)  # 7CFC00
            lemonchiffon = _c_bg_rgb(255, 250, 205)  # FFFACD
            lightblue = _c_bg_rgb(173, 216, 230)  # ADD8E6
            lightcoral = _c_bg_rgb(240, 128, 128)  # F08080
            lightcyan = _c_bg_rgb(224, 255, 255)  # E0FFFF
            lightgoldenrodyellow = _c_bg_rgb(250, 250, 210)  # FAFAD2
            lightgray = _c_bg_rgb(211, 211, 211)  # D3D3D3
            lightgreen = _c_bg_rgb(144, 238, 144)  # 90EE90
            lightgrey = _c_bg_rgb(211, 211, 211)  # D3D3D3
            lightpink = _c_bg_rgb(255, 182, 193)  # FFB6C1
            lightsalmon = _c_bg_rgb(255, 160, 122)  # FFA07A
            lightseagreen = _c_bg_rgb(32, 178, 170)  # 20B2AA
            lightskyblue = _c_bg_rgb(135, 206, 250)  # 87CEFA
            lightslategray = _c_bg_rgb(119, 136, 153)  # 778899
            lightslategrey = _c_bg_rgb(119, 136, 153)  # 778899
            lightsteelblue = _c_bg_rgb(176, 196, 222)  # B0C4DE
            lightyellow = _c_bg_rgb(255, 255, 224)  # FFFFE0
            lime = _c_bg_rgb(0, 255, 0)  # 00FF00
            limegreen = _c_bg_rgb(50, 205, 50)  # 32CD32
            linen = _c_bg_rgb(250, 240, 230)  # FAF0E6
            magenta = _c_bg_rgb(255, 0, 255)  # FF00FF
            maroon = _c_bg_rgb(128, 0, 0)  # 800000
            mediumaquamarine = _c_bg_rgb(102, 205, 170)  # 66CDAA
            mediumblue = _c_bg_rgb(0, 0, 205)  # 0000CD
            mediumorchid = _c_bg_rgb(186, 85, 211)  # BA55D3
            mediumpurple = _c_bg_rgb(147, 112, 219)  # 9370DB
            mediumseagreen = _c_bg_rgb(60, 179, 113)  # 3CB371
            mediumslateblue = _c_bg_rgb(123, 104, 238)  # 7B68EE
            mediumspringgreen = _c_bg_rgb(0, 250, 154)  # 00FA9A
            mediumturquoise = _c_bg_rgb(72, 209, 204)  # 48D1CC
            mediumvioletred = _c_bg_rgb(199, 21, 133)  # C71585
            midnightblue = _c_bg_rgb(25, 25, 112)  # 191970
            mintcream = _c_bg_rgb(245, 255, 250)  # F5FFFA
            mistyrose = _c_bg_rgb(255, 228, 225)  # FFE4E1
            moccasin = _c_bg_rgb(255, 228, 181)  # FFE4B5
            navajowhite = _c_bg_rgb(255, 222, 173)  # FFDEAD
            navy = _c_bg_rgb(0, 0, 128)  # 000080
            oldlace = _c_bg_rgb(253, 245, 230)  # FDF5E6
            olive = _c_bg_rgb(128, 128, 0)  # 808000
            olivedrab = _c_bg_rgb(107, 142, 35)  # 6B8E23
            orange = _c_bg_rgb(255, 165, 0)  # FFA500
            orangered = _c_bg_rgb(255, 69, 0)  # FF4500
            orchid = _c_bg_rgb(218, 112, 214)  # DA70D6
            palegoldenrod = _c_bg_rgb(238, 232, 170)  # EEE8AA
            palegreen = _c_bg_rgb(152, 251, 152)  # 98FB98
            paleturquoise = _c_bg_rgb(175, 238, 238)  # AFEEEE
            palevioletred = _c_bg_rgb(219, 112, 147)  # DB7093
            papayawhip = _c_bg_rgb(255, 239, 213)  # FFEFD5
            peachpuff = _c_bg_rgb(255, 218, 185)  # FFDAB9
            peru = _c_bg_rgb(205, 133, 63)  # CD853F
            pink = _c_bg_rgb(255, 192, 203)  # FFC0CB
            plum = _c_bg_rgb(221, 160, 221)  # DDA0DD
            powderblue = _c_bg_rgb(176, 224, 230)  # B0E0E6
            purple = _c_bg_rgb(128, 0, 128)  # 800080
            rebeccapurple = _c_bg_rgb(102, 51, 153)  # 663399
            red = _c_bg_rgb(255, 0, 0)  # FF0000
            rosybrown = _c_bg_rgb(188, 143, 143)  # BC8F8F
            royalblue = _c_bg_rgb(65, 105, 225)  # 4169E1
            saddlebrown = _c_bg_rgb(139, 69, 19)  # 8B4513
            salmon = _c_bg_rgb(250, 128, 114)  # FA8072
            sandybrown = _c_bg_rgb(244, 164, 96)  # F4A460
            seagreen = _c_bg_rgb(46, 139, 87)  # 2E8B57
            seashell = _c_bg_rgb(255, 245, 238)  # FFF5EE
            sienna = _c_bg_rgb(160, 82, 45)  # A0522D
            silver = _c_bg_rgb(192, 192, 192)  # C0C0C0
            skyblue = _c_bg_rgb(135, 206, 235)  # 87CEEB
            slateblue = _c_bg_rgb(106, 90, 205)  # 6A5ACD
            slategray = _c_bg_rgb(112, 128, 144)  # 708090
            slategrey = _c_bg_rgb(112, 128, 144)  # 708090
            snow = _c_bg_rgb(255, 250, 250)  # FFFAFA
            springgreen = _c_bg_rgb(0, 255, 127)  # 00FF7F
            steelblue = _c_bg_rgb(70, 130, 180)  # 4682B4
            tan = _c_bg_rgb(210, 180, 140)  # D2B48C
            teal = _c_bg_rgb(0, 128, 128)  # 008080
            thistle = _c_bg_rgb(216, 191, 216)  # D8BFD8
            tomato = _c_bg_rgb(255, 99, 71)  # FF6347
            turquoise = _c_bg_rgb(64, 224, 208)  # 40E0D0
            violet = _c_bg_rgb(238, 130, 238)  # EE82EE
            wheat = _c_bg_rgb(245, 222, 179)  # F5DEB3
            white = _c_bg_rgb(255, 255, 255)  # FFFFFF
            whitesmoke = _c_bg_rgb(245, 245, 245)  # F5F5F5
            yellow = _c_bg_rgb(255, 255, 0)  # FFFF00
            yellowgreen = _c_bg_rgb(154, 205, 50)  # 9ACD32


class Prompt:
    # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797

    del_left_line = "\x1b[1K"
    del_right_line = "\x1b[0K"
    del_line = "\r\x1b[K"
    # del_line = "\x1b[2K" # ダメ

    def __init__(self) -> None:
        self._prev_n_new_line = 0

    # def print(self, *args, **kwargs):
    #     self._prev_n_new_line = 0
    #     builtins.print(*args, **kwargs)

    # def print_update(self, small_obj, clip=True):
    #     texts = str(small_obj).split("\n")

    #     for _ in range(self._prev_n_new_line):
    #         builtins.print(Prompt.del_line + self.cursor_up(1), end="")

    #     self._prev_n_new_line = len(texts)

    #     for i, s in enumerate(texts):
    #         if clip:
    #             max_len = os.get_terminal_size().columns
    #             cnt, first_, last_ = Prompt._clip(s, max_len, max_len - 5, max_len)
    #             if cnt > max_len:
    #                 s = first_ + " ..."

    #         builtins.print(Prompt.del_line + s)

    #     # self.scroll_up(len(texts))

    @staticmethod
    def cursor_up(n: int):
        return f"\x1b[{n}A"

    @staticmethod
    def cursor_down(n: int):
        return f"\x1b[{n}B"

    @staticmethod
    def cursor_right(n: int):
        return f"\x1b[{n}C"

    @staticmethod
    def cursor_left(n: int):
        return f"\x1b[{n}D"

    @staticmethod
    def erase_left_character(n: int = 1):
        # Asked ChatGPT in English
        return f"\x1b[{n}D\x1b[K"

    @staticmethod
    def scroll_up(n: int):
        builtins.print(f"\x1b[{n}T", end="")

    @staticmethod
    def print_one_line(small_obj, clip=True):
        text = str(small_obj)

        if clip:
            text = Prompt.clip(text)

        builtins.print(Prompt.del_line + text, end="", flush=True)

    @staticmethod
    def clip(text: str):
        max_len = os.get_terminal_size().columns
        cnt, first_, last_ = Prompt._clip(text, max_len, max_len - 5, max_len)
        if cnt > max_len:
            text = first_ + " ..."
        return text

    # @staticmethod
    # def print_multi_line(obj, n: int, clip=True):
    #     texts = str(obj).split("\n")

    #     # builtins.print(Prompt.del_line)
    #     for _ in range(n):
    #         builtins.print(Prompt.del_line + Prompt.cursor_up(1), end="")

    #     for i, s in enumerate(texts[:n]):
    #         if clip:
    #             max_len = os.get_terminal_size().columns
    #             cnt, first_, last_ = Prompt._clip(s, max_len, max_len - 5, max_len)
    #             if cnt > max_len:
    #                 s = first_ + " ..."

    #         builtins.print(Prompt.del_line + s)

    #         # if i < n - 1:
    #         #     builtins.print(Prompt.del_line + s)
    #         # else:
    #         #     builtins.print(Prompt.del_line + s, end="")

    @staticmethod
    def fit_terminal(text):
        max_len = os.get_terminal_size().columns
        cnt, first_, last_ = Prompt._clip(text, max_len, 15, max_len - 20)
        if cnt <= max_len:
            return text
        else:
            return first_ + " ... " + last_

    @staticmethod
    def _clip(text, len_max, len_first_max, len_last_max):
        cnt = 0
        first_size = 0
        for i, c in enumerate(text):  # i: 0 to N-1
            if unicodedata.east_asian_width(c) in "FWA":
                cnt += 2
            else:
                cnt += 1

            if first_size == 0:
                if cnt == len_first_max:
                    first_size = i + 1
                elif cnt > len_first_max:
                    first_size = i

        last_size = 0

        if len_max <= cnt:
            cnt_ = 0
            for i, c in enumerate(reversed(text)):
                if unicodedata.east_asian_width(c) in "FWA":
                    cnt_ += 2
                else:
                    cnt_ += 1

                if last_size == 0:
                    if cnt_ == len_last_max:
                        last_size = i + 1
                        break
                    elif cnt_ > len_last_max:
                        last_size = i
                        break

            return cnt, text[:first_size], text[-last_size:]
        else:
            return cnt, None, None
