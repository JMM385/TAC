#!/usr/bin/env python
# coding: utf-8

# In[38]:


# LIBRERÍAS

# os
from os.path import join, dirname

# Pandas, Numpy
import pandas as pd
import numpy as np

# PLOTS
import matplotlib.pyplot as plt
import seaborn as sns

# SCIPY (resuelve PULS)
from scipy import interpolate
from scipy.optimize import minimize

# BOKEH (app)
from bokeh.io import output_notebook, show, curdoc
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Slider, Select, PreText, LinearAxis, Range1d

# Shapely
import shapely
from shapely.geometry import Polygon


# In[39]:


# Importo CSVs 

# Topografía
topo = pd.read_csv(join(dirname(__file__), 'data/Topografia_AyV.csv')) # A y Vol. en función de h
topo['AREA_m2'] = topo['AREA_km2'] * (1000**2)


# In[40]:


# Hidrograma de entrada variable (Hidrograma de Kozeny)
Q_base = 20.0 # m^3/s (propuesto)
def kozeny(Q_tr, tp, m, t):
    """Devuelve ordenada del hidrograma de entrada I(t) para el caudal al pico,
    tiempo al pico y factor de forma m dados"""
    q = ((t / tp) * np.exp(1 - t/tp))**m
    I_t = (Q_tr - Q_base)*q + Q_base
    return I_t


# In[42]:


# Interpolación a través de SPLINE para A=f(h) y V=f(h)

# To numpy (asi lo pide el argumento de interpolate.splrep)
cota = topo['COTA_msnm'].to_numpy() # X
area = topo['AREA_m2'].to_numpy() # Y1
volumen = topo['VOLUMEN_hm3'].to_numpy() # Y2

# Fit
area_spline = interpolate.splrep(cota, area) #type: tuple
volumen_spline = interpolate.splrep(cota, volumen) #type: tuple

# Remuestreo de COTA cada 1m
cota_resampled = np.arange(topo['COTA_msnm'].min(), topo['COTA_msnm'].max() + 1, 1) 

# Predict
area_resampled_spline = interpolate.splev(cota_resampled, area_spline, der=0)
volumen_resampled_spline = interpolate.splev(cota_resampled, volumen_spline, der=0)


# In[43]:


# Sección transversal del embalse en V variable y hasta 70 m de altura (640 a 710 msnm)
y_ascending = [i for i in range(70+1)] # Profundidades del embalse
y_descending = y_ascending[1:]
y_descending = y_descending[::-1] # Reversed
y = y_descending + y_ascending

def seccion_V(alfa):
    """Devuelve las coordenadas de la sección transversal en V con ángulo variable(en grados) para 0 a 70 m de profundidad"""
    alfa_rad = alfa *2*np.pi/360
    x = [i/np.tan(alfa_rad) for i in range(140+1)]
    coord = list(zip(x, y))
    return coord

def L_seccion_V(alfa_V, h):
    """Devuelve la longitud [m] de la sección transversal en V con ángulo variable(en grados) para 0 a 70 m de profundidad"""
    alfa_rad = alfa_V *2*np.pi/360
    L = h*2 / np.tan(alfa_rad)
    return L

s = seccion_V(30)
polygon = Polygon(s)

def area_seccion_V_transv(h, s):   
    """Devuelve el área en m2 de la sección transversal embalsepara el nivel de embalse dado (ambos en metros)"""
    """s(sección): es una lista de tuples"""
    
    s_h = [(x, y) for (x, y) in s if y <= h]
    polygon_h = Polygon(s_h)
    
    return polygon_h.area

def vol_seccion_V(h, L, s):   
    """Devuelve el volumen en hm3 del embalse para el nivel y longitud de embalse dado (ambos en metros)"""
    """s(sección): es una lista de tuples"""
    
    s_h = [(x, y) for (x, y) in s if y <= h]
    polygon_h = Polygon(s_h)
    
    return polygon_h.area*L / 100**3

df_embalse_V = pd.DataFrame()

df_embalse_V['profundidad'] = [i for i in range(1, 70+1)]
df_embalse_V['cota'] = df_embalse_V['profundidad'] + 640.0

df_embalse_V['area_transv'] = df_embalse_V.apply(lambda row: area_seccion_V_transv(row['profundidad'], s), axis=1)

df_embalse_V['area_planta'] = df_embalse_V.apply(lambda row: (7500 * L_seccion_V(30, row['profundidad'])) / 1000**2, axis=1) #km^2

df_embalse_V['vol'] = df_embalse_V.apply(lambda row: vol_seccion_V(row['profundidad'], 7500, s), axis=1)

df_embalse_V.tail()


# In[44]:


# Defino función para evaluar A y V para todo 'h' en el rango

def area_h(embalse, nivel, L_emb=0, alfa_V=0):
    """Devuelve el área del embalse original o en V para el nivel dado en m^2 """
    if embalse == 'embalse_original':
        a_1 = interpolate.splev(nivel, area_spline, der=0)
        return float(a_1)
    if embalse == 'embalse_V':
        a_2 =  L_emb * L_seccion_V(alfa_V=alfa_V, h= nivel-640)
        return a_2
    
def volumen_h(nivel):
    """Devuelve el volumen del embalse para el nivel dado en m^3 """
    v = interpolate.splev(nivel, volumen_spline, der=0)
    return v * (100**3)


# In[48]:


# Ley de VERTEDERO
c = 2.2 # m^0.5/s
B_0 = 50.0 # m (en el TP, B es incógnita / no supere un cierto nivel)
H_cresta_0 = 692.0 # msnm

def vertedero_h(nivel, B, H_cresta):
    """Devuelve el caudal erogado por vertedero libre para el nivel de embalse, altura de cresta y ancho de vertedero dados"""
   
    if nivel >= H_cresta:     
        O_h = c * B * ((nivel - H_cresta)**(3/2))
        return O_h
    
    if nivel < H_cresta:
        return 0.0
   
    else:
        return np.nan


# In[50]:


# Definición del método de laminación: RK2

k = 600 #s (se trata de un hidrograma de entrada equiespaciado en el tiempo)

# RK2
def rk_2(embalse, nivel, B, H_cresta, I_t1, I_t2, L_emb=0, alfa_V=0):
    """Se entra con un nivel inicial, devuelve el nivel final luego de un tiempo k""" 
    
    if embalse == 'embalse_original':
        # q1 se evalua en h0 y to
        q1_rk2 = k * ((I_t1 - vertedero_h(nivel, B, H_cresta)) / area_h('embalse_original', nivel)) # m

        # q2 se evalua en h0+q1 y t1 (t1-t0 = k)
        q2_rk2 = k * ((I_t2 - vertedero_h(nivel + q1_rk2, B, H_cresta)) / area_h('embalse_original', nivel + q1_rk2)) # m

        h_rk2 = nivel + (1/2)*(q1_rk2+q2_rk2) # msnm

        return h_rk2
    
    if embalse == 'embalse_V':
        # q1 se evalua en h0 y to
        q1_rk2 = k * ((I_t1 - vertedero_h(nivel, B, H_cresta)) / area_h('embalse_V', nivel, L_emb=L_emb, alfa_V=alfa_V)) # m

        # q2 se evalua en h0+q1 y t1 (t1-t0 = k)
        q2_rk2 = k * ((I_t2 - vertedero_h(nivel + q1_rk2, B, H_cresta)) / area_h('embalse_V', nivel + q1_rk2, L_emb=L_emb, alfa_V=alfa_V)) # m

        h_rk2 = nivel + (1/2)*(q1_rk2+q2_rk2) # msnm
        
        return h_rk2


# In[53]:


# Inicio df 

df_rk2 = pd.DataFrame()
df_rk2['T_h'] = [(1/6)*i for i in range(100*6 + 1)]
df_rk2['I'] = kozeny(1500, 12, 8, df_rk2['T_h'])
df_rk2['nivel_embalse'] = np.nan
df_rk2.loc[0, 'nivel_embalse'] = 691.75 #CB inicial arbitraría

df_rk2.head()


# In[55]:


# Aplico RK2
for i in range(1, len(df_rk2)-1):
    # Parámetros de la función RK_2
    n_0 = df_rk2.loc[i-1, 'nivel_embalse']
    I_i = df_rk2.loc[i,'I']
    I_ii = df_rk2.loc[i+1,'I']
    
    df_rk2.loc[i, 'nivel_embalse'] = rk_2('embalse_V', n_0, B_0, H_cresta_0, I_i, I_ii, 4000, 30) 

# Conozco los niveles en el embalse, calculo O(h)
df_rk2['O'] = df_rk2.apply(lambda row: vertedero_h(row['nivel_embalse'], B_0, H_cresta_0), axis=1)

df_rk2.tail()


# In[57]:


source_V = ColumnDataSource(data = {'profundidad' : df_embalse_V['profundidad'],
                                    'cota' : df_embalse_V['cota'],
                                    'area_transv' : df_embalse_V['area_transv'],
                                    'area_planta' : df_embalse_V['area_planta'],
                                    'vol' : df_embalse_V['vol']
                                                                 })


# In[58]:


source_rk2 = ColumnDataSource(data={
                                'T_h' : df_rk2['T_h'],
                                'I' : df_rk2['I'],
                                'nivel' : df_rk2['nivel_embalse'],
                                'O' : df_rk2['O'],
                               })


# In[59]:


# WIDGETS

# Sliders Altura de Presa, Ancho del vertedero y nivel incial del embalse
slider_1 = Slider(start=20, end=70, value=50, step=5, title='Altura de la Presa [m]') #Slider altura de presa
slider_2 = Slider(start=20, end=70, value=40, step=5, title='Nivel Inicial de Embalse [m]') #Slider nivel inicial de embalse
slider_3 = Slider(start=10, end=300, value=B_0, step=5, title='Largo del vertedero [m]') #Slider ancho de vertedero

# Sliders Hidrograma de Entrada
slider_Q = Slider(start=100, end=5000, value=1500, step=50, title='Caudal Pico de Entrada [m^3/s]') #Slider caudal pico
slider_tp = Slider(start=1, end=75, value=15, step=1, title='Tiempo al Pico de Entrada [hs]') #Slider Tiempo al pico
slider_m = Slider(start=1, end=30, value=5, step=1, title='Factor de forma') #Slider ancho de vertedero

# Sliders Embalse
slider_alfa = Slider(start=1, end=90, value=30, step=1, title='Ángulo sección en V') #Slider ángulo
slider_L = Slider(start=1, end=20, value=10, step=1, title='Longitud del embalse en V [km]') #Slider longitud del embalse #km 

# Select forma del embalse
select_embalse = Select(options=['Embalse original', 'Embalse en V'], value='Embalse original', title='Geometría del embalse')


# In[60]:


stats = PreText(text='')
def parametros(I_max, O_max, T_Imax, T_Omax, N_max):
    # Update stats
    
    
    # Caudal Imax, Omax
    a = 'Los caudales máximos de entrada y salida en [m3/s]:' + '\n' + str(round(I_max, 1)) + '\n' + str(round(O_max, 1))
    # Tiempo al pico I, O
    b = 'Los máximos ocurren en t[hs]:' + '\n' + str(round(T_Imax, 1)) + '\n' + str(round(T_Omax, 1))
    # Atenuación y desfasaje
    c = 'La atenuación de la crecida en [%] y desfasaje [hs]:' + '\n' + str(round(100*(I_max - O_max) / I_max, 1)) +'\n' + str(round(T_Omax - T_Imax, 1))
    # El nivel máximo alcanzado en el embalse
    d = 'El nivel máximo en el embalse[msnm]:' + '\n' + str(round(N_max, 1))

    stats.text = a + '\n' + b + '\n' + c + '\n' + d


# In[61]:


def callback(attr, old, new):
    
    # Slider values
    H_cresta = slider_1.value + 640
    nivel_inicial = slider_2.value + 640
    B = slider_3.value
    
    Q_p = slider_Q.value
    tp = slider_tp.value
    m = slider_m.value
    
    alfa_vble = slider_alfa.value
    L_embalse = slider_L.value
    
    # Update Área y Volumen en función de la cota
    s = seccion_V(alfa_vble)
    
    df_embalse_V['area_transv'] = df_embalse_V.apply(lambda row: area_seccion_V_transv(row['profundidad'], s), axis=1)
    
    df_embalse_V['area_planta'] = df_embalse_V.apply(lambda row: ((L_embalse*1000) * L_seccion_V(alfa_vble, row['profundidad'])) / 1000**2, axis=1) #km^2

    df_embalse_V['vol'] = df_embalse_V.apply(lambda row: vol_seccion_V(row['profundidad'], L_embalse*1000, s), axis=1)
    
    new_data_V = {'profundidad' : df_embalse_V['profundidad'],
                'cota' : df_embalse_V['cota'],
                'area_transv' : df_embalse_V['area_transv'],
                'area_planta' : df_embalse_V['area_planta'],
                'vol' : df_embalse_V['vol']
                }
    source_V.data = new_data_V

    # Update laminación       
    df_rk2['I'] = kozeny(Q_p, tp, m, df_rk2['T_h'])


    df_rk2.loc[0, 'nivel_embalse'] = nivel_inicial

    # Aplico RK2


    if select_embalse.value == 'Embalse original':
        for i in range(1, len(df_rk2)-1):
            # Parámetros de la función RK_2
            n_0 = df_rk2.loc[i-1, 'nivel_embalse']
            I_i = df_rk2.loc[i,'I']
            I_ii = df_rk2.loc[i+1,'I']
            df_rk2.loc[i, 'nivel_embalse'] = rk_2('embalse_original', n_0, B, H_cresta, I_i, I_ii)
            
        # Conozco los niveles en el embalse, calculo O(h)
        df_rk2['O'] = df_rk2.apply(lambda row: vertedero_h(row['nivel_embalse'], B, H_cresta), axis=1)
        
        new_data = {
                'T_h' : df_rk2['T_h'],
                'I' : df_rk2['I'],
                'nivel' : df_rk2['nivel_embalse'],
                'O' : df_rk2['O'],
                           }
        
        stats = parametros(df_rk2['I'].max(),df_rk2['O'].max(), df_rk2.loc[df_rk2['I'].idxmax(), 'T_h'], df_rk2.loc[df_rk2['O'].idxmax(), 'T_h'], df_rk2['nivel_embalse'].max())




    if select_embalse.value == 'Embalse en V':
        for i in range(1, len(df_rk2)-1):
            # Parámetros de la función RK_2
            n_0 = df_rk2.loc[i-1, 'nivel_embalse']
            I_i = df_rk2.loc[i,'I']
            I_ii = df_rk2.loc[i+1,'I']
            df_rk2.loc[i, 'nivel_embalse'] = rk_2('embalse_V', n_0, B, H_cresta, I_i, I_ii, L_embalse*1000, alfa_vble)

        # Conozco los niveles en el embalse, calculo O(h)
        df_rk2['O'] = df_rk2.apply(lambda row: vertedero_h(row['nivel_embalse'], B, H_cresta), axis=1)

        new_data = {
                'T_h' : df_rk2['T_h'],
                'I' : df_rk2['I'],
                'nivel' : df_rk2['nivel_embalse'],
                'O' : df_rk2['O'],
                           }
        
        stats = parametros(df_rk2['I'].max(),df_rk2['O'].max(), df_rk2.loc[df_rk2['I'].idxmax(), 'T_h'], df_rk2.loc[df_rk2['O'].idxmax(), 'T_h'], df_rk2['nivel_embalse'].max())

        
    source_rk2.data = new_data


# In[62]:


slider_1.on_change('value', callback)
slider_2.on_change('value', callback)
slider_3.on_change('value', callback)

slider_Q.on_change('value', callback)
slider_tp.on_change('value', callback)
slider_m.on_change('value', callback)

slider_alfa.on_change('value', callback)
slider_L.on_change('value', callback)

select_embalse.on_change('value', callback)


# In[65]:


# Hidrogramas
p_1 = figure(plot_width=750, plot_height=500, 
             x_axis_label='Tiempo [hs]', y_axis_label='Q [m^3/s]', title='Hidrogramas de Entrada y Salida')

p_1.line(x='T_h', y='I', source=source_rk2, color='blue', legend_label='Hidrograma de Entrada')
p_1.line(x='T_h', y='O', source=source_rk2, color='red', legend_label='Hidrograma de Salida')

p_1.background_fill_color = 'lightblue'
p_1.background_fill_alpha = 0.20

# Nivel
p_2 = figure(plot_width=750, plot_height=500, 
             x_axis_label='Tiempo [hs]', y_axis_label='Nivel [msnm]', title='Nivel de Embalse')
p_2.line(x='T_h', y='nivel', source=source_rk2, color='green')

p_2.background_fill_color = 'lightblue'
p_2.background_fill_alpha = 0.20

# Área
p_3 = figure(plot_width=750, plot_height=500, 
             x_axis_label='Cota [msnm]', y_axis_label='Área [km^2]', title='Área en función de COTA',
             )

p_3.line(x='cota', y='area_planta', source=source_V, color='purple', legend_label='Embalse en V')
p_3.line(cota_resampled, area_resampled_spline/1000**2, color='orange', legend_label='Embalse original (SPLINE)')
p_3.circle(topo['COTA_msnm'], topo['AREA_m2']/1000**2, color='blue', legend_label='Embalse original')

p_3.background_fill_color = 'lightblue'
p_3.background_fill_alpha = 0.20
p_3.legend.location = 'top_left'

# Volumen
p_4 = figure(plot_width=750, plot_height=500, 
             x_axis_label='Cota [msnm]', y_axis_label='Volumen [hm^3]', title='Volumen en función de COTA',
             )

p_4.line(x='cota', y='vol', source=source_V, color='purple',legend_label='Embalse en V')
p_4.line(cota_resampled, volumen_resampled_spline, color='orange', legend_label='Embalse original (SPLINE)')
p_4.circle(topo['COTA_msnm'], topo['VOLUMEN_hm3'], legend_label='Embalse original')

p_4.background_fill_color = 'lightblue'
p_4.background_fill_alpha = 0.20
p_4.legend.location = 'top_left'


# In[ ]:


widgets_1 = column(slider_1, slider_2, slider_3)
widgets_2 = column(slider_Q, slider_tp, slider_m)
widgets_3 = column(slider_alfa, slider_L, select_embalse)

layout = column(row(widgets_1, widgets_2, widgets_3, stats), 
                row(p_1, p_3),
                row(p_2, p_4)
               )
                

curdoc().add_root(layout)

