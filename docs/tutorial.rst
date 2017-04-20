Getting started
---------------

Inicializar cube
++++++++++++++++

Initializing is easy::

    from muse.musecube import MuseCube
    cube = MuseCube(filename_cube, filename_white)


Get a spectrum
++++++++++++++

You can get an spectrum of a geometrical region by using::

    sp1 = cube.get_spec(134, 219, 5, mode='mean')

This ``sp1`` is an ``XSpectrum1D`` object of the spaxels within a circle of radius 5 at position xy=(134, 219).




la primera función de la clase es solo para un display de
las regiones, para ver si se necesita ajustar mejor los parametros de sextractor. esta funcion es:

cube.plot_sextractor_regions()
la funcion necesita un parámetro obligatorio, el nombre del archivo output de sextractor.

Ademas se le puede dar un parametro opcional: flag_threshold, que por defecto es 16, pero puede ser recomendable 32 también, Lo que hace ese parámetro es que las regiones que tengan un flag de sextractor mayor se indican en rojo.
(La documentación de la función esta al dia en la clase)

Cuando esté satisfecho con las regiones, puede usar:

cube.save_sextractor_specs():
Esta función tiene varios parámetros, pero los únicos que le deberian importar son:
sextractor_filename: El nombre del output de SExtractor
flag_threshold= por defecto 16: Es igual que la funcion anterior, solo que en este caso, si una region tiene un flag mayor al treshold no le saca espectro.

redmonster_format= True (por defecto) Si es True, los espectros se guardan en un formato leible para redmonster, si es False, se guardan en un .fits de 2 extensiones, una con el wave y otra con el flux.

El formato del nombre de guardado es: IDsex_radec_RMF.fits


La sintaxis para correr esa funcion entonces es:
cube.save_sextractor_specs(sex_out.cat) (usando los valores default de los demás parámetros)

Una vez que tengan todos los espectros guardados en formato redmonster (_RMF.fits), lo siguiente es medirle el redshift a cada fuente identificada. De aquí en adelante tienen que tener redmonster funcionando para seguir.
Adjunte 2 scripts, plot_spec_temp.py y plot_all_specs.py. El primero corre redmonster para un espectro en formato redmonster, después lee el output de redmosnter y grafica el espectro junto a los 5 templates y z's mas probables segun redmonster. EL segundo (plot_all_specs.py) hace lo mismo para todos los espectros en formato redmonster que encuentre en ese directorio.
Por lo tanto, lo único que hay que hacer, es tener estos 2 scripts y todos los espectros obtenidos en un mismo directorio, y correr el segundo script:

python plot_all_specs.py

Con eso ya se obtiene para cada espectro una imagen que muestre los mejores 5 fits de redmonster a cada uno. El nombre de estas imágenes (en pngs) es el mismo que el nombre de los espectros, así que no hay confusiones para saber que fit corresponde a que espectro.


Además, el script plot_all_specs.py crea una tabla que se llama source_table.dat, donde está toda la información de los fits de redmonster (los z, sigma_z, y sqr_chi) además del id de redmonster, las coordenadas, el nombre y la magnitud en r de cada fuente (Para que la magnitud en r que sale ahi tenga sentido es necesario calibrar el zero_point en cada campo antes)
De aquí en adelante, lo siguiente es una inspección visual de los fits para saber a cuales creerles.


Por si acaso, acá dejo como calibrar el zeropoint de todas maneras:
1) Elegir una fuente en el campo que tenga bien medida su magnitud en r
2) Sacarle es espectro a esa fuente: Para esto lo mas facil es elegir el x_central, y_central y un radio y usar la funcion :
w,f=a.get_spec_image([xc,yc],radio)   (Esa función en particular está hecha para que xc, yc y radio estén en pixeles por comodidad) w y f contienen el espectro.

3) usar cube.calculate_mag(w,f,'r',zeropoint_mag=¿?) y ver para que zeropoint_mag, la magnitud obtenida con calculate_mag es correcta.


4) Darle ese zeropoint_mag a save_sextractor_specs() como parametro, de la forma zeropoint_mag = -9.4528492 por ejemplo... al momento de correr esa función, en el segundo paso de este tutorial.

Eso sería todo lo importante creo, es harto  texto pero en verdad son 3 lineas de codigo. En resumen:

cube.plot_sextractor_regions() Para ver las regiones
cube.save_sextractor_specs() Para guardar los espectros
python plot_all_specs.py para correr sextractor, guardar los outputs y crear una tabla