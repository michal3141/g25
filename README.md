# g25
mixture.py: A utility for modeling ancestry using G25 coordinates, admixture calculators coordinates etc.

Dependencies: Linux, Python3, cvxpy, numpy

Usage examples:

Regular example:

    (base) michal3141@Lenovo:~/admixture/global25/g25$ python mixture.py Global25_PCA_modern_scaled.txt michal3141.txt 
    -------------- ANCESTRY BREAKDOWN: -------------
    Ukrainian                                         --->	21.240%
    Latvian                                           --->	19.058%
    Lithuanian_RA                                     --->	16.176%
    German                                            --->	12.495%
    Russian_Kostroma                                  --->	11.760%
    Macedonian                                        --->	5.556%
    Lithuanian_SZ                                     --->	3.590%
    Greek_Central_Macedonia                           --->	3.074%
    Sardinian                                         --->	2.155%
    Brahmin_Uttar_Pradesh                             --->	1.433%
    Polish                                            --->	1.156%
    Ket                                               --->	0.963%
    Yemenite_Amran                                    --->	0.774%
    Swedish                                           --->	0.509%
    Ju_hoan_North                                     --->	0.061%
    ------------------------------------------------
    Fit error: 0.00608282997975319

With penalty:

	(base) michal3141@Lenovo:~/admixture/global25/g25$ python mixture.py Global25_PCA_modern_scaled.txt michal3141.txt pen=0.01
	-------------- ANCESTRY BREAKDOWN: -------------
	Ukrainian                                         --->	44.195%
	Russian_Kursk                                     --->	34.612%
	German                                            --->	9.224%
	Mordovian                                         --->	9.217%
	Russian_Tver                                      --->	2.683%
	Moldovan_o                                        --->	0.068%
	------------------------------------------------
	Fit error: 0.00971654404360762

At least 90% of Polish ancestry (you could also use equal, and less than comparisons):

	(base) michal3141@Lenovo:~/admixture/global25/g25$ python mixture.py Global25_PCA_modern_scaled.txt michal3141.txt 'Polish>=0.90'
	-------------- ANCESTRY BREAKDOWN: -------------
	Polish                                            --->	90.000%
	Ukrainian                                         --->	3.927%
	Karelian                                          --->	3.152%
	Ket                                               --->	0.914%
	Brahmin_Uttar_Pradesh                             --->	0.910%
	Mbuti                                             --->	0.429%
	Nganassan                                         --->	0.402%
	Greenlander_East                                  --->	0.125%
	BedouinB                                          --->	0.099%
	Ju_hoan_North                                     --->	0.041%
	------------------------------------------------
	Fit error: 0.012352986384727531

At least 95% of Polish ancestry and 0% of Ukrainian:

	(base) michal3141@Lenovo:~/admixture/global25/g25$ python mixture.py Global25_PCA_modern_scaled.txt michal3141.txt 'Polish>=0.95' 'Ukrainian=0'
	-------------- ANCESTRY BREAKDOWN: -------------
	Polish                                            --->	95.000%
	Karelian                                          --->	1.976%
	Ket                                               --->	1.055%
	Brahmin_Uttar_Pradesh                             --->	0.766%
	Mbuti                                             --->	0.452%
	Nganassan                                         --->	0.392%
	Eskimo_Naukan                                     --->	0.193%
	BedouinB                                          --->	0.121%
	Ju_hoan_North                                     --->	0.034%
	Greenlander_East                                  --->	0.011%
	------------------------------------------------
	Fit error: 0.013038959615808695

At least 98% of Polish and Slovakian ancestry together!:

	(base) michal3141@Lenovo:~/admixture/global25/g25$ python mixture.py Global25_PCA_modern_scaled.txt michal3141.txt 'Polish+Slovakian>=0.98'
	-------------- ANCESTRY BREAKDOWN: -------------
	Polish                                            --->	65.408%
	Slovakian                                         --->	32.592%
	Nganassan                                         --->	1.092%
	Ket                                               --->	0.673%
	Mbuti                                             --->	0.189%
	Ju_hoan_North                                     --->	0.046%
	------------------------------------------------
	Fit error: 0.012895470113353954

At least 98% of Polish and Slovakian ancestry together with additional penalty:

	(base) michal3141@Lenovo:~/admixture/global25/g25$ python mixture.py Global25_PCA_modern_scaled.txt michal3141.txt 'Polish+Slovakian>=0.98' pen=0.01
	-------------- ANCESTRY BREAKDOWN: -------------
	Polish                                            --->	52.074%
	Slovakian                                         --->	45.926%
	Ket                                               --->	1.408%
	Russian_Kostroma                                  --->	0.592%
	------------------------------------------------
	Fit error: 0.014263306424617072

Run with the limited number of populations with non-zero admix.
Example below is using a sheet with all population averages from Davidski and we
demand to have at most 2 populations (count=2) with non-zero admix!
The run below took about 40 seconds on my machine.
You may experiment with this option but remember that the running time
grows exponentially with the number of populations/individuals in your reference sheet.

	(base) michal3141@Lenovo:~/admixture/global25/g25$ time python mixture.py Global25_PCA_pop_averages_scaled.txt michal3141.txt count=2
	-------------- ANCESTRY BREAKDOWN: -------------
	Baltic_EST_IA                                     --->	69.951%
	UKR_Cimmerian_o                                   --->	30.049%
	------------------------------------------------
	Fit error: 0.018940013697675166





