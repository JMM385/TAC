#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Librerías
from os.path import join, dirname
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.optimize import minimize
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.io import curdoc
from bokeh.layouts import column, row, widgetbox
from bokeh.models import ColumnDataSource, Slider, Select, DataTable, DateFormatter, TableColumn, CustomJS, PreText
from bokeh.plotting import figure


# In[3]:

topo = pd.read_csv(join(dirname(__file__), 'data/Topografia_AyV.csv'))
I_t = pd.read_csv(join(dirname(__file__), 'data/HidrogramaDeEntrada.csv'))

# topo = pd.read_csv('Topografia_AyV.csv')
# I_t = pd.read_csv('HidrogramaDeEntrada.csv')


# In[4]:


topo['AREA_m2'] = topo['AREA_km2'] * (1000**2)

# Interpolación a través de SPLINE para A=f(h) y V=f(h)

cota = topo['COTA_msnm'].to_numpy() # X
area = topo['AREA_m2'].to_numpy() # Y1
volumen = topo['VOLUMEN_hm3'].to_numpy() # Y2

area_spline = interpolate.splrep(cota, area) #type: tuple
volumen_spline = interpolate.splrep(cota, volumen) #type: tuple


cota_resampled = np.arange(topo['COTA_msnm'].min(), topo['COTA_msnm'].max() + 1, 1) # Remuestreo de COTA cada 1m

area_resampled_spline = interpolate.splev(cota_resampled, area_spline, der=0)
volumen_resampled_spline = interpolate.splev(cota_resampled, volumen_spline, der=0)


# In[5]:


# Defino función para evaluar A y V para cualquier 'h'

def area_h(nivel):
    
    a = interpolate.splev(nivel, area_spline, der=0)
    return a
    

def volumen_h(nivel):
    
    v = interpolate.splev(nivel, volumen_spline, der=0)
    return v * (100**3)


# In[6]:


# Ley de VERTEDERO
# Se podrían hacer variar y analizar los cambios en el hidrograma de salida
c = 2.2 # m^0.5/s
B = 50.0 # m (en el TP, B es incógnita)
H_cresta = 692.0 # msnm

def vertedero_h(nivel):
    if nivel >= H_cresta:
        # Devuelve el caudal erogado por vertedero libre para el nivel de embalse dado
        O_h = c * B * ((nivel - H_cresta)**(3/2))
        return O_h
    else:
        return 0.0


# In[7]:


# Definición del método de laminación: Euler, RK2, RK3, RK4, Puls

k = 600 #s (se trata de un hidrograma de entrada equiespaciado en el tiempo)

# Euler
def euler(nivel, I_t1):
    
    q1_euler = k * ((I_t1 - vertedero_h(nivel)) / area_h(nivel)) # m
    
    h_euler = nivel + q1_euler # msnm
    
    return h_euler


# RK2
def rk_2(nivel, I_t1, I_t2):
    # Se entra con un nivel inicial, devuelve el nivel final luego de un tiempo k
    
    # q1 se evalua en h0 y to
    q1_rk2 = k * ((I_t1 - vertedero_h(nivel)) / area_h(nivel)) # m
    
    # q2 se evalua en h0+q1 y t1 (t1-t0 = k)
    q2_rk2 = k * ((I_t2 - vertedero_h(nivel + q1_rk2)) / area_h(nivel + q1_rk2)) # m
              
    h_rk2 = nivel + (1/2)*(q1_rk2+q2_rk2) # msnm
    
    return h_rk2

# RK3
def rk_3(nivel, I_t1, I_t2, I_t3):
    
    q1_rk3 = k * ((I_t1 - vertedero_h(nivel)) / area_h(nivel)) # m
    
    q2_rk3 = k * ((I_t2 - vertedero_h(nivel + q1_rk3)) / area_h(nivel + q1_rk3)) # m
              
    q3_rk3 = k * ((I_t3 - vertedero_h(nivel + q1_rk3 + q2_rk3)) / area_h(nivel + q1_rk3 + q2_rk3)) # m
    
    h_rk3 = nivel + (1/3)*(q1_rk3+q2_rk3+q3_rk3) # msnm
    
    return h_rk3

# RK4
def rk_4(nivel, I_t1, I_t2, I_t3, I_t4):
    
    q1_rk4 = k * ((I_t1 - vertedero_h(nivel)) / area_h(nivel)) # m
    
    q2_rk4 = k * ((I_t2 - vertedero_h(nivel + q1_rk4)) / area_h(nivel + q1_rk4)) # m
     
    q3_rk4 = k * ((I_t3 - vertedero_h(nivel + q1_rk4 + q2_rk4)) / area_h(nivel + q1_rk4 + q2_rk4)) # m
    
    q4_rk4 = k * ((I_t4 - vertedero_h(nivel + q1_rk4 + q2_rk4 +q3_rk4)) / area_h(nivel + q1_rk4 + q2_rk4 + q3_rk4)) # m
    
    h_rk4 = nivel + (1/4)*(q1_rk4+q2_rk4+q3_rk4+q4_rk4) # msnm
    
    return h_rk4

# PULS MODIFICADO
def puls_modificado(h_i, I_i, I_ii):
    # Función Puls Modificado, devuelve h_ii / resuelve la siguiente ecuación:
    def funcion_objetivo(h_ii):
        
        termino_1 = 2 * (volumen_h(h_ii) - volumen_h(h_i)) / k
        termino_2 = I_i + I_ii
        termino_3 = vertedero_h(h_i) + vertedero_h(h_ii)
        
        objetivo = abs(termino_1 - termino_2 + termino_3)
        return objetivo
    
    
    sol = minimize(funcion_objetivo, 692.0, method = 'SLSQP', bounds = [(600, 800)])
    return float(sol.x)


# In[8]:


df = pd.DataFrame({
    't' : I_t['Tiempo_seg'],
    'I' : I_t['I_m3/s'],
    'nivel_embalse_euler' : np.nan,
    'nivel_embalse_rk2' : np.nan,
    'nivel_embalse_rk3' : np.nan,
    'nivel_embalse_rk4' : np.nan,
    'nivel_embalse_puls' : np.nan,
                        })


# In[9]:


nivel_inicial = 680
df.loc[0, ['nivel_embalse_euler', 'nivel_embalse_rk2', 'nivel_embalse_rk3', 'nivel_embalse_rk4', 'nivel_embalse_puls' ]] = nivel_inicial
df.head()


# In[10]:


for i in range(1, len(df)):
    n_0 = df.loc[i-1, 'nivel_embalse_euler']
    I_i = df.loc[i, 'I']
    df.loc[i, 'nivel_embalse_euler'] = euler(n_0, I_i)

df['O_euler'] = df.apply(lambda row: vertedero_h(row['nivel_embalse_euler']), axis=1)
df.tail()


# In[11]:


# RK2
for i in range(1, len(df)-1):
    # Parámetros de la función RK_2
    nivel_inicial = df.loc[i-1, 'nivel_embalse_rk2']
    I_i = df.loc[i,'I']
    I_ii = df.loc[i+1,'I']
    
    df.loc[i, 'nivel_embalse_rk2'] = rk_2(nivel_inicial, I_i, I_ii) 

df['O_rk2'] = df.apply(lambda row: vertedero_h(row['nivel_embalse_rk2']), axis=1)
df.tail()


# In[12]:


# RK3
for i in range(1, len(df)-2):
    # Parámetros de la función RK_3
    nivel_inicial = df.loc[i-1, 'nivel_embalse_rk3']
    I_i = df.loc[i,'I']
    I_ii = df.loc[i+1,'I']
    I_iii = df.loc[i+2,'I']
    
    df.loc[i, 'nivel_embalse_rk3'] = rk_3(nivel_inicial, I_i, I_ii, I_iii)

df['O_rk3'] = df.apply(lambda row: vertedero_h(row['nivel_embalse_rk3']), axis=1)
df.tail()


# In[13]:


# RK4
for i in range(1, len(df)-3):
    # Parámetros de la función RK_4
    nivel_inicial_rk4 = df.loc[i-1, 'nivel_embalse_rk4']
    I_i_rk4 = df.loc[i,'I']
    I_ii_rk4 = df.loc[i+1,'I']
    I_iii_rk4 = df.loc[i+2,'I']
    I_iv_rk4 = df.loc[i+3,'I']
    
    df.loc[i, 'nivel_embalse_rk4'] = rk_4(nivel_inicial_rk4, I_i_rk4, I_ii_rk4, I_iii_rk4, I_iv_rk4)
    
df['O_rk4'] = df.apply(lambda row: vertedero_h(row['nivel_embalse_rk4']), axis=1)
df.tail()


# In[14]:


# Puls Modificado:
I_list = df['I'].tolist()
h = []
h.append(df.loc[0, 'nivel_embalse_puls'])
for i in range(1, len(I_list) - 1) :
    nivel_inicial = h[i-1] 
    h.append(puls_modificado(nivel_inicial, I_list[i-1], I_list[i]))

df.loc[:len(df) - 2, 'nivel_embalse_puls'] = h

df['O_puls'] = df.apply(lambda row: vertedero_h(row['nivel_embalse_puls']), axis=1)
df.tail()


# In[15]:


# Volumen de embalse para cada ti
for nivel in ['nivel_embalse_euler', 'nivel_embalse_rk2','nivel_embalse_rk3', 'nivel_embalse_rk4', 'nivel_embalse_puls']:
    for volumen in ['vol_euler', 'vol_rk2','vol_rk3', 'vol_rk4', 'vol_puls']:
        df[volumen] = volumen_h(df[nivel]) / 100**3

# Para cada fila se consideró el nivel correspondiente a la misma (se podría tomar el promedio entre un mes y el siguiente)
df['T_h'] = [(1/6)*i for i in range(len(df))]
df.head()


# In[25]:


source = ColumnDataSource(data={'t' : df['t'],
                                'T_h' : df['T_h'],
                                'I' : df['I'],
                                'nivel' : df['nivel_embalse_euler'],
                                'O' : df['O_euler'],
                                'vol' : df['vol_euler'],
                                
                               })


# In[26]:


# Sliders
slider_1 = Slider(start=20, end=70, value=50, step=5, title='Altura de la Presa [m]') #Slider altura de presa
slider_2 = Slider(start=20, end=70, value=40, step=5, title='Nivel Inicial de Embalse [m]') #Slider nivel inicial de embalse
slider_3 = Slider(start=10, end=300, value=B, step=5, title='Largo del vertedero [m]') #Slider ancho de vertedero


# In[27]:


# Select Métodos
select_metodo = Select(options=['Euler', 'RK2', 'RK3', 'RK4', 'Puls Modificado'], value='Euler', title='Método')


# In[28]:


# Update stats
stats = PreText(text='', width=500)
# Caudal Imax, Omax
a = 'Los caudales máximos de entrada y salida en [m3/s]:' + '\n' + str(round(df['I'].max(), 1)) + '\n' + str(round(df['O_euler'].max(), 1))
# Tiempo al pico I, O
b = 'Los máximos ocurren en t[hs]:' + '\n' + str(round(df.loc[df['I'].idxmax(), 'T_h'], 1)) + '\n' + str(round(df.loc[df['O_euler'].idxmax(), 'T_h'], 1))
# Atenuación y desfasaje
c = 'La atenuación de la crecida en [%] y desfasaje [hs]:' + '\n' + str(round(100*(df['I'].max() - df['O_euler'].max()) / df['I'].max(), 1)) +'\n' + str(round(df.loc[df['O_euler'].idxmax(), 'T_h'] - df.loc[df['I'].idxmax(), 'T_h'], 1))
# El nivel máximo alcanzado en el embalse
d = 'El nivel máximo en el embalse[msnm]:' + '\n' + str(round(df['nivel_embalse_euler'].max(), 1))

stats.text = a + '\n' + b + '\n' + c + '\n' + d


# In[33]:


def callback(attr, old, new):
    
    # Ley de VERTEDERO
    c = 2.2 # m^0.5/s
    B = slider_3.value
    H_cresta = 640 + slider_1.value  # msnm

    def vertedero_h(nivel):
        if nivel >= H_cresta:
            # Devuelve el caudal erogado por vertedero libre para el nivel de embalse dado
            O_h = c * B * ((nivel - H_cresta)**(3/2))
            return O_h
        else:
            return 0.0
        
    # Euler
    def euler(nivel, I_t1):

        q1_euler = k * ((I_t1 - vertedero_h(nivel)) / area_h(nivel)) # m

        h_euler = nivel + q1_euler # msnm

        return h_euler


    # RK2
    def rk_2(nivel, I_t1, I_t2):
        # Se entra con un nivel inicial, devuelve el nivel final luego de un tiempo k

        # q1 se evalua en h0 y to
        q1_rk2 = k * ((I_t1 - vertedero_h(nivel)) / area_h(nivel)) # m

        # q2 se evalua en h0+q1 y t1 (t1-t0 = k)
        q2_rk2 = k * ((I_t2 - vertedero_h(nivel + q1_rk2)) / area_h(nivel + q1_rk2)) # m

        h_rk2 = nivel + (1/2)*(q1_rk2+q2_rk2) # msnm

        return h_rk2

    # RK3
    def rk_3(nivel, I_t1, I_t2, I_t3):

        q1_rk3 = k * ((I_t1 - vertedero_h(nivel)) / area_h(nivel)) # m

        q2_rk3 = k * ((I_t2 - vertedero_h(nivel + q1_rk3)) / area_h(nivel + q1_rk3)) # m

        q3_rk3 = k * ((I_t3 - vertedero_h(nivel + q1_rk3 + q2_rk3)) / area_h(nivel + q1_rk3 + q2_rk3)) # m

        h_rk3 = nivel + (1/3)*(q1_rk3+q2_rk3+q3_rk3) # msnm

        return h_rk3

    # RK4
    def rk_4(nivel, I_t1, I_t2, I_t3, I_t4):

        q1_rk4 = k * ((I_t1 - vertedero_h(nivel)) / area_h(nivel)) # m

        q2_rk4 = k * ((I_t2 - vertedero_h(nivel + q1_rk4)) / area_h(nivel + q1_rk4)) # m

        q3_rk4 = k * ((I_t3 - vertedero_h(nivel + q1_rk4 + q2_rk4)) / area_h(nivel + q1_rk4 + q2_rk4)) # m

        q4_rk4 = k * ((I_t4 - vertedero_h(nivel + q1_rk4 + q2_rk4 +q3_rk4)) / area_h(nivel + q1_rk4 + q2_rk4 + q3_rk4)) # m

        h_rk4 = nivel + (1/4)*(q1_rk4+q2_rk4+q3_rk4+q4_rk4) # msnm

        return h_rk4

    # PULS MODIFICADO
    def puls_modificado(h_i, I_i, I_ii):
        # Función Puls Modificado, devuelve h_ii / resuelve la siguiente ecuación:
        def funcion_objetivo(h_ii):

            termino_1 = 2 * (volumen_h(h_ii) - volumen_h(h_i)) / k
            termino_2 = I_i + I_ii
            termino_3 = vertedero_h(h_i) + vertedero_h(h_ii)

            objetivo = abs(termino_1 - termino_2 + termino_3)
            return objetivo

        sol = minimize(funcion_objetivo, 692.0, method = 'SLSQP', bounds = [(600, 800)])
        return float(sol.x)

    
    #slider_2.end = slider_1.value
    #slider_2 = Slider(start=20, end=slider_1.value, value=40, step=5, title='Nivel Inicial de Embalse [m]') #Slider nivel inicial de embalse
    
    df = pd.DataFrame({
    't' : I_t['Tiempo_seg'],
    'I' : I_t['I_m3/s'],
    'nivel_embalse_euler' : np.nan,
    'nivel_embalse_rk2' : np.nan,
    'nivel_embalse_rk3' : np.nan,
    'nivel_embalse_rk4' : np.nan,
    'nivel_embalse_puls' : np.nan,
                        })
    
    nivel_inicial = slider_2.value + 640
    df.loc[0, ['nivel_embalse_euler', 'nivel_embalse_rk2', 'nivel_embalse_rk3', 'nivel_embalse_rk4', 'nivel_embalse_puls' ]] = nivel_inicial
    
    # Aplico EULER
    for i in range(1, len(df)):
        n_0 = df.loc[i-1, 'nivel_embalse_euler']
        I_i = df.loc[i, 'I']
        df.loc[i, 'nivel_embalse_euler'] = euler(n_0, I_i)

    df['O_euler'] = df.apply(lambda row: vertedero_h(row['nivel_embalse_euler']), axis=1)
    
    #Aplico RK2
    for i in range(1, len(df)-1):
        # Parámetros de la función RK_2
        nivel_inicial = df.loc[i-1, 'nivel_embalse_rk2']
        I_i = df.loc[i,'I']
        I_ii = df.loc[i+1,'I']
        df.loc[i, 'nivel_embalse_rk2'] = rk_2(nivel_inicial, I_i, I_ii) 

    df['O_rk2'] = df.apply(lambda row: vertedero_h(row['nivel_embalse_rk2']), axis=1)
    
    #Aplico RK3
    for i in range(1, len(df)-2):
        # Parámetros de la función RK_3
        nivel_inicial = df.loc[i-1, 'nivel_embalse_rk3']
        I_i = df.loc[i,'I']
        I_ii = df.loc[i+1,'I']
        I_iii = df.loc[i+2,'I']
        df.loc[i, 'nivel_embalse_rk3'] = rk_3(nivel_inicial, I_i, I_ii, I_iii)

    df['O_rk3'] = df.apply(lambda row: vertedero_h(row['nivel_embalse_rk3']), axis=1)
    
    #Aplico RK4
    for i in range(1, len(df)-3):
        # Parámetros de la función RK_4
        nivel_inicial_rk4 = df.loc[i-1, 'nivel_embalse_rk4']
        I_i_rk4 = df.loc[i,'I']
        I_ii_rk4 = df.loc[i+1,'I']
        I_iii_rk4 = df.loc[i+2,'I']
        I_iv_rk4 = df.loc[i+3,'I']
        df.loc[i, 'nivel_embalse_rk4'] = rk_4(nivel_inicial_rk4, I_i_rk4, I_ii_rk4, I_iii_rk4, I_iv_rk4)
    
    df['O_rk4'] = df.apply(lambda row: vertedero_h(row['nivel_embalse_rk4']), axis=1)
    
    #Aplico PULS
    
    I_list = df['I'].tolist()
    h = []
    h.append(df.loc[0, 'nivel_embalse_puls'])
    for i in range(1, len(I_list) - 1) :
        nivel_inicial = h[i-1] 
        h.append(puls_modificado(nivel_inicial, I_list[i-1], I_list[i]))

    df.loc[:len(df) - 2, 'nivel_embalse_puls'] = h

    df['O_puls'] = df.apply(lambda row: vertedero_h(row['nivel_embalse_puls']), axis=1)
    
    # Volumen de embalse para cada ti
    for nivel in ['nivel_embalse_euler', 'nivel_embalse_rk2','nivel_embalse_rk3', 'nivel_embalse_rk4', 'nivel_embalse_puls']:
        for volumen in ['vol_euler', 'vol_rk2','vol_rk3', 'vol_rk4', 'vol_puls']:
            df[volumen] = volumen_h(df[nivel]) / 100**3
    
    df['T_h'] = [(1/6)*i for i in range(len(df))]
    
  
    if select_metodo.value == 'Euler':
        new_data = {'t' : df['t'],
                    'T_h' : df['T_h'],
                    'I' : df['I'],
                    'nivel' : df['nivel_embalse_euler'],
                    'O' : df['O_euler'],
                    'vol' : df['vol_euler'],
                   }
        # Caudal Imax, Omax
        a = 'Los caudales máximos de entrada y salida en [m3/s]:' + '\n' + str(round(df['I'].max(), 1)) + '\n' + str(round(df['O_euler'].max(), 1))
        # Tiempo al pico I, O
        b = 'Los máximos ocurren en t[hs]:' + '\n' + str(round(df.loc[df['I'].idxmax(), 'T_h'], 1)) + '\n' + str(round(df.loc[df['O_euler'].idxmax(), 'T_h'], 1))
        # Atenuación y desfasaje
        c = 'La atenuación de la crecida en [%] y desfasaje [hs]:' + '\n' + str(round(100*(df['I'].max() - df['O_euler'].max()) / df['I'].max(), 1)) +'\n' + str(round(df.loc[df['O_euler'].idxmax(), 'T_h'] - df.loc[df['I'].idxmax(), 'T_h'], 1))
        # El nivel máximo alcanzado en el embalse
        d = 'El nivel máximo en el embalse[msnm]:' + '\n' + str(round(df['nivel_embalse_euler'].max(), 1))

        stats.text = a + '\n' + b + '\n' + c + '\n' + d
        
        
    if select_metodo.value == 'RK2':
        new_data = {'t' : df['t'],
                    'T_h' : df['T_h'],
                    'I' : df['I'],
                    'nivel' : df['nivel_embalse_rk2'],
                    'O' : df['O_rk2'],
                    'vol' : df['vol_rk2'],
                   }
             # Caudal Imax, Omax
        a = 'Los caudales máximos de entrada y salida en [m3/s]:' + '\n' + str(round(df['I'].max(), 1)) + '\n' + str(round(df['O_rk2'].max(), 1))
        # Tiempo al pico I, O
        b = 'Los máximos ocurren en t[hs]:' + '\n' + str(round(df.loc[df['I'].idxmax(), 'T_h'], 1)) + '\n' + str(round(df.loc[df['O_rk2'].idxmax(), 'T_h'], 1))
        # Atenuación y desfasaje
        c = 'La atenuación de la crecida en [%] y desfasaje [hs]:' + '\n' + str(round(100*(df['I'].max() - df['O_rk2'].max()) / df['I'].max(), 1)) +'\n' + str(round(df.loc[df['O_rk2'].idxmax(), 'T_h'] - df.loc[df['I'].idxmax(), 'T_h'], 1))
        # El nivel máximo alcanzado en el embalse
        d = 'El nivel máximo en el embalse[msnm]:' + '\n' + str(round(df['nivel_embalse_rk2'].max(), 1))

        stats.text = a + '\n' + b + '\n' + c + '\n' + d
    
    if select_metodo.value == 'RK3':
        new_data = {'t' : df['t'],
                    'T_h' : df['T_h'],
                    'I' : df['I'],
                    'nivel' : df['nivel_embalse_rk3'],
                    'O' : df['O_rk3'],
                    'vol' : df['vol_rk3'],
                   }
          # Caudal Imax, Omax
        a = 'Los caudales máximos de entrada y salida en [m3/s]:' + '\n' + str(round(df['I'].max(), 1)) + '\n' + str(round(df['O_rk3'].max(), 1))
        # Tiempo al pico I, O
        b = 'Los máximos ocurren en t[hs]:' + '\n' + str(round(df.loc[df['I'].idxmax(), 'T_h'], 1)) + '\n' + str(round(df.loc[df['O_rk3'].idxmax(), 'T_h'], 1))
        # Atenuación y desfasaje
        c = 'La atenuación de la crecida en [%] y desfasaje [hs]:' + '\n' + str(round(100*(df['I'].max() - df['O_rk3'].max()) / df['I'].max(), 1)) +'\n' + str(round(df.loc[df['O_rk3'].idxmax(), 'T_h'] - df.loc[df['I'].idxmax(), 'T_h'], 1))
        # El nivel máximo alcanzado en el embalse
        d = 'El nivel máximo en el embalse[msnm]:' + '\n' + str(round(df['nivel_embalse_rk3'].max(), 1))
    
    if select_metodo.value == 'RK4':
        new_data = {'t' : df['t'],
                    'T_h' : df['T_h'],
                    'I' : df['I'],
                    'nivel' : df['nivel_embalse_rk4'],
                    'O' : df['O_rk4'],
                    'vol' : df['vol_rk4'],
                   }
                    
          # Caudal Imax, Omax
        a = 'Los caudales máximos de entrada y salida en [m3/s]:' + '\n' + str(round(df['I'].max(), 1)) + '\n' + str(round(df['O_rk4'].max(), 1))
        # Tiempo al pico I, O
        b = 'Los máximos ocurren en t[hs]:' + '\n' + str(round(df.loc[df['I'].idxmax(), 'T_h'], 1)) + '\n' + str(round(df.loc[df['O_rk4'].idxmax(), 'T_h'], 1))
        # Atenuación y desfasaje
        c = 'La atenuación de la crecida en [%] y desfasaje [hs]:' + '\n' + str(round(100*(df['I'].max() - df['O_rk4'].max()) / df['I'].max(), 1)) +'\n' + str(round(df.loc[df['O_rk4'].idxmax(), 'T_h'] - df.loc[df['I'].idxmax(), 'T_h'], 1))
        # El nivel máximo alcanzado en el embalse
        d = 'El nivel máximo en el embalse[msnm]:' + '\n' + str(round(df['nivel_embalse_rk4'].max(), 1))
    
    if select_metodo.value == 'Puls Modificado':
        new_data = {'t' : df['t'],
                    'T_h' : df['T_h'],
                    'I' : df['I'],
                    'nivel' : df['nivel_embalse_puls'],
                    'O' : df['O_puls'],
                    'vol' : df['vol_puls'],
                   }
        # Caudal Imax, Omax
        a = 'Los caudales máximos de entrada y salida en [m3/s]:' + '\n' + str(round(df['I'].max(), 1)) + '\n' + str(round(df['O_puls'].max(), 1))
        # Tiempo al pico I, O
        b = 'Los máximos ocurren en t[hs]:' + '\n' + str(round(df.loc[df['I'].idxmax(), 'T_h'], 1)) + '\n' + str(round(df.loc[df['O_puls'].idxmax(), 'T_h'], 1))
        # Atenuación y desfasaje
        c = 'La atenuación de la crecida en [%] y desfasaje [hs]:' + '\n' + str(round(100*(df['I'].max() - df['O_puls'].max()) / df['I'].max(), 1)) +'\n' + str(round(df.loc[df['O_puls'].idxmax(), 'T_h'] - df.loc[df['I'].idxmax(), 'T_h'], 1))
        # El nivel máximo alcanzado en el embalse
        d = 'El nivel máximo en el embalse[msnm]:' + '\n' + str(round(df['nivel_embalse_puls'].max(), 1))
        
    source.data = new_data


# In[39]:


slider_1.on_change('value', callback)
slider_2.on_change('value', callback)
slider_3.on_change('value', callback)
select_metodo.on_change('value', callback)


# In[35]:


p_1 = figure(plot_width=1000, plot_height=500, x_axis_label='Tiempo [hs]', y_axis_label='Q [m^3/s]', title='Hidrogramas de Entrada y Salida')
p_1.line(x='T_h', y='I', source=source, color='blue', legend_label='Hidrograma de Entrada')
p_1.line(x='T_h', y='O', source=source, color='red', legend_label='Hidrograma de Salida')

p_2 = figure(plot_width=500, plot_height=300, x_axis_label='Tiempo [hs]', y_axis_label='Nivel [msnm]', title='Nivel de Embalse')
p_2.line(x='T_h', y='nivel', source=source, color='green')

p_3 = figure(plot_width=500, plot_height=300, x_axis_label='Tiempo [hs]', y_axis_label='Volumen [hm3]', title='Volumen de Embalse')
p_3.line(x='T_h', y='vol', source=source, color='pink')


p_4 = figure(plot_width=500, plot_height=300, x_axis_label='Cota [msnm]', y_axis_label='Área [has]', title='Área f(h)')
p_4.line(cota_resampled, area_resampled_spline/10000, color='orange')
p_4.circle(topo['COTA_msnm'], topo['AREA_m2']/10000)


p_5 = figure(plot_width=500, plot_height=300, x_axis_label='Cota [msnm]', y_axis_label='Volumen [hm^3]', title='Volumen f(h)')
p_5.line(cota_resampled, volumen_resampled_spline, color='orange')
p_5.circle(topo['COTA_msnm'], topo['VOLUMEN_hm3'])


# In[36]:


layout = column(row(column(select_metodo, slider_1, slider_2, slider_3), stats), 
                row(p_1),
                row(p_2, p_4),
                row(p_3, p_5))

curdoc().add_root(layout)

