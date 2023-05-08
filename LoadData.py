import pandas as pd

DROP_CYL = False #DON'T CHANGE

def load_data(path):
    #Read the file
    df = pd.read_csv(path)

    # Drop unneeded columns
    df = df.drop('Trim', axis=1)
    df = df.drop('Engine', axis=1)
    df = df.drop('yearsold', axis = 1)
    df = df.drop('ID', axis = 1)

    #Possible: Drop num cylinders
    if (DROP_CYL):
        df = df.drop('NumCylinders', axis = 1)

    # Clean up drive type (drop outlier DriveTypes)
    df = df[df['DriveType'].str.len() == 3]
    df['DriveType'] = df['DriveType'].str.lower()
    drive_types = df['DriveType'].value_counts().gt(10)
    df = df.loc[df['DriveType'].isin(drive_types[drive_types].index)]

    #Clean up BodyType (Drop outliers with freq. under 50)
    df['BodyType'] = df['BodyType'].str.lower()
    body_types = df['BodyType'].value_counts().gt(50)
    df = df.loc[df['BodyType'].isin(body_types[body_types].index)]

    #Clean up model (Drop outliers with freq. under 20)
    df['Model'] = df['Model'].str.lower()
    models = df['Model'].value_counts().gt(20)
    df = df.loc[df['Model'].isin(models[models].index)]

    #Clean up Make (Drop outliers with freq. under 20)
    df['Make'] = df['Make'].str.lower()
    makes = df['Make'].value_counts().gt(20)
    df = df.loc[df['Make'].isin(makes[makes].index)]

    #Clean up numerical values
    df = df[df['Mileage'] > 100]
    df = df[df['Mileage'] < 400000]
    df = df[df['Year'] > 1920]
    df = df[df['Year'] < 2021]
    df = df[df['pricesold'] > 200]

    #Clean up num-cylinders if not dropped
    if (DROP_CYL == False):
        df = df[ (df['NumCylinders'] != 0) | (df['Make'] == 'tesla')]

    # Clean up zipcode and drop nulls
    ndf = df[df['zipcode'].str.len() == 5]
    ndf = ndf[ndf['zipcode'].str[2:5] != '***']
    ndf['zipcode'] = ndf['zipcode'].str[:-2]
    df['zipcode'] = pd.to_numeric(ndf['zipcode'], downcast='integer').astype('Int64')
    df = df.dropna() #Drop nulls
    df['zipcode'] = df['zipcode'].astype('int64')

    return df