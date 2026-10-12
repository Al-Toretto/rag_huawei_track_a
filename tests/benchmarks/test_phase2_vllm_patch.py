# SPDX-License-Identifier: Apache-2.0
"""Test the public patch command against the exact offline vLLM 0.10.0 source."""

# Standard
from pathlib import Path
import base64
import hashlib
import importlib.util
import subprocess
import sys
import zlib

# Third Party
import pytest

HELPER = Path(__file__).resolve().parents[2] / "tools/phase2/apply_vllm_patch.py"
ORIGINAL_SHA256 = "37c953ac5801d1647327f0b13a61831893a75c97ebc65e7e122f7968817fd8d0"
PATCHED_SHA256 = "eb5ea8148852adfba5e165fbd4af8b1a10501bae7b5883022c3c180a0d15371e"
BACKUP_SUFFIX = ".phase2-vllm-0.10.0.bak"

# Exact Apache-2.0 fixture from vllm-project/vllm tag v0.10.0:
# vllm/v1/worker/gpu_worker.py; Copyright contributors to the vLLM project.
# Compressing the source keeps the test standalone without executing the worker.
UPSTREAM_FIXTURE = (
    b"c%0omYj5L5lHdI+dNs(cw2(QE#qI41tKb}ulNj;Xh9AiWvja|xY*}1WB$p3+l)?P>tEz62{h}n#Bmr&(3`QcmtE=l>"
    b")ek>o*O#yVad?(QBCEvVX)LljStg>Kv7>@VD{=TLcyaJdw)!?n#c^J2%H(cU--r+OjCDU*lxKCBESfqmD^};MUJ3Sa"
    b"cJ_u9W&SS_)d!=|=!kuHdCMO2@?MlIN_kZU@bn;A7kP;gicS0dE^5E!)xomN*90GiOp^suZ;lpKUGk_tmT%thjNgfJ"
    b"%CaU+RcqO`-V{l8*LJ)4`SK(@{{H0nhtu=#rtB!&OxZ<ICwa!xgLYJyMYJ-"
    b"01aVTy35wW!k!6fmEX%si4{5p%MD|eObNpYTL5~5kWzymDBRo2$Ct4G2ptju@xYLv(T<5V!!-"
    b"AJQP2o$DC3V8n<QEZ757^cZHVrk%A>fsidA0_I@9JBLI*gjC&etJ~8<rw&A~Eq}9Na&Iz?H080*fu|wO3C;UH`0jWq"
    b"`RW^QIU=rK3|-"
    b"^IEjn+`(*$LiU`}uUh@mz@+)zT|WULo_u7qDS1$ee>bA4+m>hf)zOuFq_vc66CXr`>e04Evcf!KAZsN+sQUIr%Ijrb"
    b"uB*0b)RZMKDr{fqZSy<@=_aA*J6y`AH@v#n+E$2J7ImaJ1tlrhB2IujH^3xNU9ELfCus-pyX2d2v8hEhWpBhfFE`g2"
    b"FRE2uPuZG&hoI-"
    b"n1P9eXJiG)wxJ$Bb#48~c0L0;S&nZQ*ss8$kXZJUoLYUBi_Ys;UkRL#O$@{td@Z&MQklHkDe_j_NG9lzmT{Lw!uah^"
    b"IH$R7OE?%FUg;%%d=O<U;#qG`I?G4NmxpviLnJ6!)lL^QmPbu)fu&KZVk+@Z6V`=k(yP{D#ToT-lx|g@wXx^^!zi`2"
    b";ZpuH=qi^8J!GRLkxh|{YFwFQ`gy95C)6z72Gg7)C5mi6aoX0S9O`-"
    b"v++TnZCJyF)n1O`|~c`1SlI1np}w_UICS(A1&Jvbmope8f!=QNoe5ZG8OSxELG49A_Rs6@K#4F`@IO3|5VMzwE{<`G"
    b"X(+wNyTA<g1>9_dz+hFyz#mB%v%kNZ|h6~<-"
    b"q04$N5nXv`v95CQpo>oG(Yo|a6Hw8>@5_D71aGq=5w^+k|67&A+miE+gN1K7k&Ig`vfn*b8Za<jx!~khT1}ZU=6A$V"
    b"Vd<=M(bzTd27K@oN*fW;Fiy2G#FB{f5V8$Qvr1g{N<;f?rg#a@tSm$NXEa|`otQ^gh3``uZmZ+*Asg=>w-"
    b"(;*!?iu^0S%Pz_Sj8W}hbvsFREWZm-yo!CAvFsZveOJ$64h@})>M2UmF*pf*2ho!C^s}yX?8(N*-"
    b"6G1DU1g$g>_~$hCmE_+uaHS5%$1g01d>C5TJ!O7gxvMhnH6u-"
    b"=3a<`FnkOHGf|4diA`{pCf9|G0GA;?=Yqb2Kt&zqZx!Pjv+yUFa!<NHXsHJY0QEoTjt}@W$R_2KTS7aHwd(1k4c&Wt"
    b"rgMhe8#@2M#J(tgxSRCL2POp0n(Bq9nIC3u`>#UrO5-xo~Y(;y(n0KF4dPK*>4jZU*7t0-"
    b"`Vl@^^x22o%`r9saNQ#qWk$x*$B5mo++x7g2$V=HL!FFi=3CSYPJHwgEuZQwvl_vno5Lsza#}=G3m)Kz>LneGUGhYM"
    b"DHU=m&W85SSlVwN@g9Jy_~Sauc>hlP74^t10+8f*n&w#*B3Pq(6k2OxPfTrsC{>!^%p|ML!ne}smh&%L`}>CfD>YTN"
    b"s>Ex`{JFhz1JY7lq4y=Z_Nau&R$vO0+dd^n9tcOlPY9GVPMUm%HTr&a4G1sI%r%RzG3h%Q}_i@(SY}gQXu1a;xq-"
    b"+srghj2hvN#0({yHWA)U`Sx0}s#3XtkNeGe(k*fxv=a}StzFcCw64v~#nvX}10@G5k0BmAV?<ENqtD6G!RueakqkIX"
    b"mrKA$t;F_n~_-a6iB?xHmhsO?wM5hA-"
    b"`p`p>qaO}H>uFI{0;E;P;OjYiF=ZnetTPHsfhiH_a)mwufk+txTN;eCK&dyj2Cqifgg-"
    b"J>j>EnRUM(>%PzeAGW_GaNLx2Dg6dGxe4VYvAtkDsEuH_g8AdMhen@O+vkNjSQO{@FJ|5=yOzfB>KL2vr*ow69efi?"
    b"IN;cbyew2dI*3Nj)z2)d1sY};ZYvhkpbPd?vl?5*81$1ng^*amwHhdf@7q3lk~YaU=}C>);JTO8sYHTAgfA$#{p%P-"
    b"b^vk;`zZ8i(Re5k5OV>8*<@c~9ssz7q)uTjl(rV)->0$>KkGRjd?SUW-Smr5>vo-Vx#bC@Yw)Xvnpd{lOc8vxs42HB"
    b"+X1VMnM@`nw&)roipF-"
    b"f|?AaCSvji!R6HOG>JLYzV3g<&!TiB&K%XQ4O@Gw*T4fJBo5)XX{j+ZLJhV6@_+MzB)UO_{aLdb#VXU^1mpCM;{f!4"
    b"yrn0F>@4D?N2S*g7&>du$wunXH{C(K1IN-1P@LdPa87{((8aP)a6EZQ@G4l3N8?{tD_%AxN~)-"
    b";V4&(5c4;eOZ#lIrs_46v?-c@k}x-"
    b"Kgf8MH5s@d8y+;OHdzFTQkML}(Q&XM2Qor#u>!9ZaS(}$nyqzr3fbpfnLk?4xNHd|S%BQ-OQs<QNjoN+$_-"
    b"ZvzRJun7e{5U9P|>54p;mk0qL|}UWt@bp}49VCt+UIby3ZpKL@~31HxZyB<b?<?s@$jX?v)|9;SSlCQDH#YjFq)bVx"
    b"+24(t3-qcgTXlv8>B@^3Hy`+xpy447Aesw%%7$(-"
    b")|`1mY5`tjoQb$E4hd~x+UyuP_QIeK$FLT@~JIXbW~Ht11_nk_c$if>vLh!2q<$6itT0_5ifB>PR0#u&y7U=tL1F&;"
    b"@=u75s14o|MGF0R7wkIrA8ot}ROaR|i)uIG@_*ad$|ke4KtvyarIFT{MZO^a2t4Y1y;>4E|`F-"
    b"Z`Dnnm|vvLsEcngY#t6~?F$W4C5P&&_e~B5;>!Dx%u5*jfZD)<wNhCTC*vpoWN~#l98zN>zJZ8j*^(uYpu%b$INdsw"
    b"$a-GgiRoZ0v#xDD-Wc{AoY$>zXR6xJqM=l^&(ujw|zwPa(G&{_lRwU5AI^?wqB3y@)wm%vhN>Sv+1C!7Y2`=1(P0AS"
    b"wBgr;Ru%%X}b-UXH#+qo$AoE2RX~?+uru@kbcdM_X-flTXiwx)1f%IWzf$Gn*hX;B8Ats|K+F%?9I!){!ITRf5$iTo"
    b"7!HYR{i7GuoHMpX_|ZH!qu^YBm9TEu@Lo@f!lC;|(50nR<O<7?pq_%fT;nkqB6S7vQ!SLyE{1I<WPSkSs^%poGe5w9"
    b"<hlV#YoWdRyIaJ!7X`NJ4H*UyRh240}^cqM;Ti2G$qHc8EJw>a@Ky?fJ+#Jh?PI1Zl4-y65AVc0?&$@aSG-"
    b"u};iu@L+O@kKqzuPc>`p&tuu~v5^l#3IX$OqFM$bl)lM~&5YFaTyAEqEP-"
    b"t@AB<h|?DENRRIR~8%2D`k2Og(O6(ZqM7$cWX#o8%Tt_idRvRIyB$$godj@w;Ja|B4?@{E0Z`j0m!<4xY&^CVjFoc$"
    b"NuJo4=BZ-"
    b"1H0*e&P_70F*GRl)0Mg%hW=K=2qNGo_yx<tum)(yfwz0VCUZv5${}E~=$66Oma4wQjIY`6kE~X7)gHEfj$SXgs2-"
    b"_sQAO_01^+BbVXz@zL2yczYS19o?QEe?Ov}2&1DG9tArxncbeLVy@EVV-rvL)X4O;iRc*5^&eWDr70l8ZjFaZRq_W>"
    b"mI+vMecj?H+q@e}K@0>;h#f3#Opp^qQcY=y-oFujticsp|0@G25L{8>YC2I+1v-"
    b"dwEO1pl$siej;7LmPyTqkOACilH@X<&uYp5in5d7ZEG(c8+r3V@9hG47wk*%9(#X4xJBWBBxD0dn04?i;UnnbG{5~7"
    b"QXH&xK<^bNdHYxT5}vP{Z~cJ;()ayaXj_giGd2QYg0TgNoe;bl{n)R^-"
    b"`4JIknYpDe=5D3)6cqUl~i{d67*Q796_jBGbU_48cbt3I!N4MS&8$G}s^Y~wloB?)t?++v1Gul{z^bT}}b>9c0YEOe"
    b"wpxsCIwj)XN);G4}M<a67y}=2s)C*t?5bcqdG2mILtSD~Q>y1o%X)BgmGI0`E4V*4ufvHyZ_6|4da6?4TYLnC<HSam"
    b"dzdytsC`WdjLA)muOtRIiAhoWVwBDu4UF#;(Gad?rd(H#xO|XHiwyTGF+^iAr{Nm<h{Fvuea}QqUd$0;HU95dIYaxX"
    b"kX=_D)z=BrvDv&P?97=4Q#8kuppO)Pb=aSMi32{@l{X5q5oKuCj9cj(i_WbQK+=${#D)%Z^*$rid!^@Eh91RWMEB`<"
    b"Nw}W-ev1J14fU{7R{cUtcR7hCyq7Yas09)kHzgqE9=Kxjbc!E)q1q)*pD;IcNp+eFA#-"
    b"!40qvA?e*OQThgIo^g^#c`lrc7>*!_g}+s5oV9oId=()WQOjR5{sIiADy-"
    b"wqp5X)=xyQcJ{ug+e6jKygx1fr!m^0tr=OK*`Rv8Qp-rOSTvTw@X5)nTT7Y_J$6{4kkyFQH@BAk_YlG<^NH2~7$<9-"
    b"_~&gB%xNSyG#s^j={?_$>FD?ja#h30C<xglX*9&cK>9cg1FG(ARfwpMW%{^J+jRGCW{>3BWEbYdTN=kj4m0ZPdbNAL"
    b"jNN~p01-DIOI4C;Y(zuS1i<*X^EtMG(-!6bfCZa<mpVr6{72iW5d(X&^}W~r!Tk#)#0K?vmn+vKqeXRW@l2DWC|?(-"
    b"gn1c4n6!qBW_#-"
    b"PjA1j#byx)>RMLNQFwp3Z!N_7DC(F(eFKV>h)NZN)SU_|Qlr7u!?nlgXAr4chP?)<AH7?~AqQn9MvP*{`0&uX{jDj-"
    b"$;V^5~hYL($X=~FxxHBA;@L4t4Jv=Q+0meoT-e4LqSviZjE430E@-"
    b"5~gA^XsxtHvT+$<1rezC8hUtOQ9G=F+AI2X`yf2b$#voQTmD2K%5W?hiM2&7|d0sJJBI8nE7J1pf)xc%KyP<nrtrwi"
    b"3iuTsnNj1*Pndb;rX}L^%Z8GQ@_opiH73D$b?lO9kp6;;}xfDfEUEw-"
    b"R!zlSwMWDI6<F!L57<0b`1$Q2Ads%2)AdO5Z$1c0*%ALYHD+Y(AyAhN4Ip5|c0Du{&H#a;axV5$mT`xA?liO<XuZkX"
    b"+u{W)*s<Z3Z|~pbN$|*pMPP4HyAI(v)fuOxA2FD(&zOtjMb>LC*@IULpF;qD}z6EcqQ4DrKNWOKE}j8sqz!T$-"
    b"ZQz%=c4(7ZS1`X0;RMN`7SLMC3o@2u898S|^86<r9NLj*BqDeJ`V{W<0PK<ATwjA1+pk}6DjRm&~T27iDOA+FCt)Pg"
    b"z)vLvIJs@;g)xiQtSf`&Wmq1!<yQx%e_s+<+3Uq{>Uvr6JvWb%Lq=ogW|cm^~6r{97MX^3~`I4L=qxlT_%nN00++xS"
    b"<XOdnT;?ZFOu?yfWq&H?)YM{p_zCCpJ>kG>VC($`@E8AHuoKG<8EgiMOD%8iUxNl=$8w?%V#a7QlabI_TnCKb_O^>r"
    b"~b=~AY}$&8;=d2b_MD`Lp9;0^8j-"
    b"ZRKG3}FjXt<r#&w4APab+45;Y3G`DkZ0eJdI<2I<x6=6q_y5g@wOM`Ui0PYju{(S2Rl^KLNn<$<2grav^gc(VWhCN<"
    b"Z6?zrYk53uI;00rN{UU0QDSva>d#j1#HoS0uf@$<JSY+>~Y7@QOS;T{eancGST!vos02p!83Va6Z>h4RfBZUQ0Y#}s"
    b"pE;BaM5a_v&4F3EYP(lirUpN_J=teL69#>kOy#CO$L)IN0yOB-Ssniz{#n4S5kG!k0O<i)z*FqbBqd*G}tTwsr~M|9"
    b"i#EtB@MPCz&U%5IQq6$DkISEf^R>#hzns#MfLB?OPO5VKd=_$6di4*EH9Of+cP47sRd~XR<%M`j&#8|1zFQQM4iBgD"
    b"xNXd@iOrkb}~MA&dw%*BDXUxZtAmMsnK1^7$ajEBXgPlvAY4If9DQ8A3|y`C@qa#DC&&?r9-"
    b"<1rR_5wlyb+gI(rXwSZL<kqbstY0-8nJH@Z`7R)*3BDc$J{@X2Q0OhdU_6xx-ovi0Kav7M2Tb#&vqsYQ3<gc^*u6(#"
    b"KUd12K+iZW@pBXrH~#U|vCmO*9%J~F5gJ<l2kE=eyo^AUCqF*FZxk|FJd8BH|b-"
    b"r8;%I8h`#RBPB8hrZNc@;DChxN&>D1-^!=3OUoy9Mc%c!97BS4yF#xSb1IN55gHXiTfR<dyb?-"
    b"QZTe}FwCKMTId5o+FKZ8pNL26t^qBQ?AO4&1LLAC_vx)5b*%^a`>eg|fL2s_`p7r9lat9_8(hUobF`_DSk`~6Xgh}y"
    b"tV+bRed?H1bjL<$g+9(7vvw^$1%E4R4ZHRXs}C9(y~p;m5qll27wuvC7J$lCg%(Nqw~ZJTkhdr3>5~X@#CPl(HJ{G>"
    b"eJBK-"
    b"w#LO9vd#WYS@;>X=PrRvv@`>R3Ig5kARghG7lky><^d>0OnE+7`$0ZR`}udBi1KXF;Y+hrimposkm1O8V7z;N^+<iP"
    b"t-"
    b"U8YL?JLzBO^%O)&X>pmvZ!hmDVQ*m#w|4RuBZFVj$<;@wrR$1;_K}$X_~lKF2vd8II1<x&Fhu9^Y^J7G|OIO9l4Cky"
    b"ETurUF#w52J~#&+0^%+Hm7;$?|Bnqq!!fTyknBqKs3;|4|yRPPHk8Z;D?*-"
    b"Tyym@{V$>N3D13tVbc;M`o8jbno|6t+KyMB(ySApNh53`?zZBT_yfC#_szqZEWi@V|5;1W?~R|J|r<@U-"
    b"pn*NMDeeJ9#;;c~1)RS9W#OMkzo$<Mz`Tx<?mc@*niEJQuwLKb)LghR3&8S10E;;nmUk52i?LpN*6wPpx-"
    b"P^gd%oA#ET29p<G5`hwX>2zv^~8k7xhXevHOI=aqJ8;qRE=MRV72ULHZi>HUj-niKBK>hl!$pPf%Crn)vvdaKy{V5~"
    b"C;<hxqts_LUOO+xm==@rrT0bjVE_=j`+;A#7non`{rqM(*9vio4iSAcX3IKs?@#}SyC363gI(Zv{50z!ji3bbo7NV4"
    b"Iu&T7A6X3u|`M{6Rx7Y|seW{Q?);9up<nSqm6ozgudT~~<<#P%R)Yrt9Ny?p-{cSNi2w<mt5riRKh7^WDdvz1-"
    b"1aIfoGp=8|#-"
    b"%8EjzLNYt?~8vzJ2j7z@OU5O{29Pqrm*f3Yl%)5UpWY+Q+1ehS+52$S;cM5v0qy+N_%j3#01B4BD;NpRZJ-YmwgIpw"
    b"1C^&pdlFy0iY1u6I6IDn>HlY#nC^BUfDg5+1Diy{BGPOB_S1;*8gbfvqqNMf-"
    b"{MKJk8iNPR3^=hknt*JK9|{F5K1jPAL0tN3@r;G31{j<LmRQzfSIY!~a#4drQAz|J$^KI_w3+}}Z{bG+w`zCxfK&?b"
    b"ksFW(u^Z09Q%uHz<)c~)yUxB}!w_V8(_w&;OBNZ|`0(PVXpsI?5^WEb}0pIh(U%17y?<+9EAt;1E%jY#`_=*aE7T|}"
    b"+_{OYp-(FVZzC@LE3=&*aheNh5IA_#WI!*kvrkA5vm%f#^fzI5PEgA-"
    b"O|G=DMq90jvq21zZ}mUA3Xnk^#?8|g1mUax$9PJ)h~p(Nc&%;_@0$#1ICo|$=6bibHxsHoV(8H$Y5NiZ8dJln}J<R3"
    b"foWjCGsrZ?SFEa$_ooRu_}UizYmwmVf*;NPa=Fm$bOrQNck#kapg*pIu^GSGQC@G7#0kY{msc0@NQ`aXx=4TD%(R;Q"
    b"0G?Di_7_tcB}*jID+^L=X<U>}T_sDq{>qZ$=k5#2xFkVg{-7H{<AO$Vrva0}VT=DFN`t1<IkRg3HIZ*N{-{B*vha-q"
    b"zr&abcMgU0t61EW^sjIAWUU*hUXwXJw(Pf;U)ItS{Ne)kZuy;bOUJjmY7^jA5|-"
    b"a+dRv};5e_FPV(WDil?R(tS@hiCX9AFHW}wI(uz*uHdfk<d!Tcr#c?<z(F2fyQrW+Z$esYTW}2k8m@Sw!VUw&8gV6Z"
    b"-%=@WjU#9hYQ?ByR8;}%LUEd)yi@T*EoYSdegyiUj;O4Ym;Bcgk1{=NC@W%(qXRtH($21pSNH7*ES!z%ckS)I}rOzs"
    b"k=i!Bl}wF(I8#jN$wcjJLKGXGR65EbXQ;PFee0k>)qAX_0)q`Qx6NDb@Mg6JA5hMs2zDE&v+YC7DXvg%%aw4G}`k@k"
    b"hZYsChxmyY~DXGFh&#M*;+`)t2}>>fFN!AGY4ZEm{!+ddz$bebFGni%DRg{2;R;=-"
    b"QeJ*b{`s1%f5!rhE*hmNbxQL1Dc82z~5yUn-"
    b"^}7RByeWO5X7s%ll?j4omk_NlALfj<GY{9JrFl@LB*;uCT4#`+`k+U>K}Oa0|TRwSM`RnY<4C-"
    b"+acdPv5+J@yc*dx%&qb0^C*EUVg(1zDUxf-i-hH!U_WVcer!V26ETk;H3kx>KH&5y76**e{5o3y#=^y<9m155svw~p"
    b"d3-EZFk`v)kk^zS_5USnEQAB<GCcv=p@j-Zvb@D-g&P6HPKm)fYKfu*hdY@C+ps~-M3$jzEPY<vd|@v-"
    b"7a@La|5zewPy&Ok=+dg%Mo77S2H};U9-OiF9N(lqGTUi=6xdR{HRUslYNMru)=#M@B)hl(C-"
    b"jb_sc9cSeZ_fD5()$QFm8OcmTa8L4OZGIG!G7$7iQgDr1Webn;@x4i9DFX~n^FjV1^G2i*l*q5"
)


def run_command(worker: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI against a fixture without importing the serving environment.

    Args:
        worker (Path): Offline worker fixture.
        arguments (str): Additional command-line arguments.

    Returns:
        subprocess.CompletedProcess[str]: Exit status and captured text output.
    """
    return subprocess.run(
        [sys.executable, str(HELPER), "--worker-file", str(worker), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def worker(tmp_path: Path) -> Path:
    """Return an exact upstream worker fixture inside a temporary directory.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        Path: Newly written worker source with the verified upstream hash.
    """
    source = zlib.decompress(base64.b85decode(UPSTREAM_FIXTURE))
    assert hashlib.sha256(source).hexdigest() == ORIGINAL_SHA256
    path = tmp_path / "gpu_worker.py"
    path.write_bytes(source)
    path.chmod(0o640)
    return path


def test_check_is_applicable_and_does_not_write(worker: Path) -> None:
    """A check recognizes the original source without changing directory contents."""
    before = worker.stat()
    original = worker.read_bytes()
    result = run_command(worker, "--check")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ready:")
    assert worker.read_bytes() == original
    assert worker.stat().st_mtime_ns == before.st_mtime_ns
    assert worker.stat().st_mode == before.st_mode
    assert list(worker.parent.iterdir()) == [worker]


def test_apply_matches_tested_source_and_preserves_backup(worker: Path) -> None:
    """Application produces the exact tested source and saves the original bytes."""
    original = worker.read_bytes()
    result = run_command(worker)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("applied:")
    assert hashlib.sha256(worker.read_bytes()).hexdigest() == PATCHED_SHA256
    assert worker.stat().st_mode & 0o777 == 0o640
    backup = worker.with_name(worker.name + BACKUP_SUFFIX)
    assert backup.read_bytes() == original
    assert backup.stat().st_mode & 0o777 == 0o640
    assert set(worker.parent.iterdir()) == {worker, backup}


def test_second_application_and_check_are_idempotent(worker: Path) -> None:
    """Already-applied calls leave both the worker and its backup unchanged."""
    assert run_command(worker).returncode == 0
    backup = worker.with_name(worker.name + BACKUP_SUFFIX)
    before = (worker.read_bytes(), backup.read_bytes())
    times = (worker.stat().st_mtime_ns, backup.stat().st_mtime_ns)
    for arguments in [(), ("--check",)]:
        result = run_command(worker, *arguments)
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("already-applied:")
    assert (worker.read_bytes(), backup.read_bytes()) == before
    assert (worker.stat().st_mtime_ns, backup.stat().st_mtime_ns) == times
    assert set(worker.parent.iterdir()) == {worker, backup}


@pytest.mark.parametrize("check", [False, True])
def test_changed_source_is_rejected_without_mutation(worker: Path, check: bool) -> None:
    """Changed upstream source is rejected in both check and application modes."""
    changed = worker.read_bytes() + b"\n# An unrelated local modification.\n"
    worker.write_bytes(changed)
    result = run_command(worker, *("--check",) if check else ())
    assert result.returncode == 1
    assert "Unexpected worker SHA256" in result.stderr
    assert worker.read_bytes() == changed
    assert list(worker.parent.iterdir()) == [worker]


def test_unsupported_version_is_rejected_without_mutation(worker: Path) -> None:
    """A version assertion cannot bypass the exact supported-version requirement."""
    original = worker.read_bytes()
    result = run_command(worker, "--vllm-version", "0.17.1")
    assert result.returncode == 1
    assert "Expected vLLM 0.10.0; found 0.17.1" in result.stderr
    assert worker.read_bytes() == original
    assert list(worker.parent.iterdir()) == [worker]


def test_unrelated_backup_is_preserved_and_application_rejected(worker: Path) -> None:
    """An unrelated existing backup is never overwritten and prevents replacement."""
    original = worker.read_bytes()
    backup = worker.with_name(worker.name + BACKUP_SUFFIX)
    backup.write_bytes(b"Another backup that must be retained.\n")
    result = run_command(worker)
    assert result.returncode == 1
    assert "Refusing to overwrite unrelated backup" in result.stderr
    assert worker.read_bytes() == original
    assert backup.read_bytes() == b"Another backup that must be retained.\n"
    assert set(worker.parent.iterdir()) == {worker, backup}


def test_matching_backup_can_be_reused(worker: Path) -> None:
    """An exact existing original backup supports safe retry after restoration."""
    backup = worker.with_name(worker.name + BACKUP_SUFFIX)
    backup.write_bytes(worker.read_bytes())
    timestamp = backup.stat().st_mtime_ns
    result = run_command(worker)
    assert result.returncode == 0, result.stderr
    assert hashlib.sha256(worker.read_bytes()).hexdigest() == PATCHED_SHA256
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == ORIGINAL_SHA256
    assert backup.stat().st_mtime_ns == timestamp


def test_missing_worker_returns_error_without_creating_files(tmp_path: Path) -> None:
    """A missing explicit fixture produces a clear failure without creating files."""
    result = run_command(tmp_path / "missing.py", "--check")
    assert result.returncode == 1
    assert "ERROR:" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_installed_worker_uses_distribution_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed worker discovery uses metadata without importing the GPU stack."""
    site = tmp_path / "site-packages"
    worker_path = site / "vllm/v1/worker/gpu_worker.py"
    worker_path.parent.mkdir(parents=True)
    metadata_dir = site / "vllm-0.10.0.dist-info"
    metadata_dir.mkdir()
    (metadata_dir / "METADATA").write_text("Name: vllm\nVersion: 0.10.0\n")
    monkeypatch.syspath_prepend(str(site))
    spec = importlib.util.spec_from_file_location("phase2_vllm_patch", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.installed_worker() == (worker_path, "0.10.0")
