import pandas as pd

# Funcion para eliminar corchetes
def elimina_corchete (dat):
     df_sin_corchetes = dat.applymap(lambda x: str(x).replace('[', '').replace(']', '') if isinstance(x, str) else x)
     return df_sin_corchetes

# Funcion para filtrar por la fecha y la hora del data frame
def filtra_fecha_hora (dat):
     dtime = r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}'
     temp = dat['Time'].str.contains(dtime)
     dat = dat[temp]
     return dat

# Funcion para serpara la hora de la fecha
def separa_fecha_hora (dat):
    dat['date_time'] = pd.to_datetime(dat['Time'], format='%d/%b/%Y:%H:%M:%S')
    return dat

# Ordena el data frame por IP y el tiempo
def ordena (dat):
     dat = dat.sort_values(['IP', 'date_time'])
     return dat

# Funcion para crear la sesionizacion
def sesionizacion (dat, dt):
     intervalo_sesion = pd.Timedelta(minutes=dt)
     dat['diferencia_tiempo'] = dat.groupby('IP')['date_time'].diff()
     dat['session'] = ((dat['diferencia_tiempo'] > intervalo_sesion) | dat['diferencia_tiempo'].isnull()).cumsum()
     dat.drop('diferencia_tiempo', axis=1, inplace=True)
     return dat

# Funcion para separar la URL
def separa_url (dat):
     dat = dat.str.split(' ', expand=True)
     return dat

# Limpia los recursos de datos nulos como los "?"
def limpia_recurso (dat):
     dat = dat.str.split('?', expand=True)[0]
     return dat