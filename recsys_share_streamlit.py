import base64

import csv
import ipywidgets as widgets
import io
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ppscore as pps
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import sklearn
import seaborn as sns
import urllib

from matplotlib import cm
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from urllib.request import urlopen
from io import BytesIO
from io import StringIO
from sklearn.model_selection import GridSearchCV


st.set_option('deprecation.showPyplotGlobalUse', False)




file2 = ('https://raw.githubusercontent.com/Bubami/ISCM_images/main/FinaleExporte.csv')
file1 = "C:/Users/micha/Documents/Jupiter/ISCM/Erste Versüche/FinaleExporte.csv"
df = pd.read_csv(file2, engine ='python', error_bad_lines=True, sep='[,;]')
df_all = df

#df = df.drop(df.columns[0], axis=1)
df = df.drop('Bezeichnung', axis=1)
df = df.drop('Quelle', axis=1)
df = df.drop('Anbieter', axis=1)
df = df.drop('Reifegrad', axis=1)


df.columns = [x.encode("utf-8").decode("ascii", "ignore") for x in df.columns]
df_all.columns = [x.encode("utf-8").decode("ascii", "ignore") for x in df_all.columns]


df_all = df_all.dropna(how='all')
df = df.dropna(how='all')
# User list comprehension to create a list of lists from Dataframe rows
# Speichern der .csv Datei in Eine Liste
list_of_rows = [list(row) for row in df.values]
list_of_rows_all = [list(row) for row in df_all.values]
from pandas import DataFrame

df_all = DataFrame(list_of_rows_all,
                   columns=['ID','Beschreibung','Quelle', 'Reifegrad', 'IoT', 'CPS', 'CC', 'ML', 'DLT', 'MBS','Anbieter', 'Technologie', 'Maturity_Index', 'Strategische_Ziele',
                        'Branche', 'Funktion', 'SCOR1', 'Unterprozess'])

df = DataFrame(list_of_rows,
               columns=['ID','IoT', 'CPS', 'CC', 'ML', 'DLT', 'MBS', 'Technologie', 'Maturity_Index', 'Strategische_Ziele',
                        'Branche', 'Funktion', 'SCOR1', 'Unterprozess'])


def main():

    # Register your pages
    pages = {
        "Startseite": page_third,
        "Deskriptiv": page_first,
        "Präskriptiv": page_second,
    }

    st.sidebar.title("Auswahl der Funktion")

    # Widget to select your page, you can choose between radio buttons or a selectbox
    page = st.sidebar.radio("Wählen Sie die gewünschte Funktion", tuple(pages.keys()))
    # page = st.sidebar.selectbox("Select your page", tuple(pages.keys()))

    # Display the selected page
    pages[page]()

##########################################################################################
Attribute = 6
X_test1 = []
for i in range(Attribute):
    X_test1.append([])
X_test2 = []
for i in range(Attribute):
    X_test2.append([])
##########################################################################################

#Eingabe Funktion für die Problemstellung
def eingabe():
        # Maturity Index
        st.write("""
            ### Maturity Index
            """)
        # st.selectbox erzeugt eine Auswahlleiste, in der alle Maturity Index Werte angezeigt werden. Die Funktion unique_sorted_values sortiert alle doppelt vorkommenden Werte raus.
        Maturity_Index = st.selectbox("Wählen Sie einen Maturity Index", unique_sorted_values_plus_ALL(df.Maturity_Index))
        X_test1[0] = Maturity_Index

        if st.checkbox("Zusatzinformationen zum Maturity Index anzeigen"):
            maturity_info()


        # Strategische Ziele   (ohne Zusatzinfos)
        st.write("""
            ### Strategische Ziele
            """)
        Strategische_Ziele = st.selectbox("Wählen Sie ein Strategisches Ziel",
                                          unique_sorted_values_plus_ALL(df.Strategische_Ziele))
        X_test1[1] = Strategische_Ziele


        # Branche   (ohne Zusatzinfos)
        st.write("""
            ### Branche
            """)
        Branche = st.selectbox("Wählen Sie eine Branche", unique_values_plus_ALL(df.Branche))
        X_test1[2] = Branche

        # Funktion
        st.write("""
            ### Funktion
            """)
        Funktion = st.selectbox("Wählen Sie eine Funktion", unique_sorted_values_plus_ALL(df.Funktion))
        X_test1[3] = Funktion

        if st.checkbox("Zusatzinformationen zur Funktion anzeigen"):
            funku_info()


        # SCOR 1
        st.write("""
        ### SCOR 1
        """)
        SCOR1 = st.selectbox("Wählen Sie einen SCOR1 Prozess", unique_sorted_values_plus_ALL(df.SCOR1))
        X_test1[4] = SCOR1
        if st.checkbox("Zusatzinformationen zum SCOR1 Prozess Index anzeigen"):
            scor_info()

        #Unterprozess
        st.write("""
        ### Unterprozess
        """)
        Unterprozess = st.selectbox("Wählen Sie einen Unterprozess", unique_values_plus_ALL(df.Unterprozess))
        X_test1[5] = Unterprozess
        if st.checkbox("Zusatzinformationen zum Unterprozess anzeigen"):
            up_info()

#Funktion zum aussortieren doppelter Werte
def unique_values_plus_ALL(array):
    item_layout = widgets.Layout(width='80%')
    ALL = 'ALL'
    unique = array.unique().tolist()
    unique.insert(0, ALL)
    return unique
def unique_sorted_values_plus_ALL(array):
    item_layout = widgets.Layout(width='80%')
    ALL = 'ALL'
    unique = array.unique().tolist()
    unique.sort()
    unique.insert(0, ALL)
    return unique


#Startseite
def page_third():
    markdown_schmal()
    st.title("Empfehlungsdienst zum Einsatz von 4.0 Technologien in der Suppply Chain")
    st.write("Diese Website soll die deskriptiven und präskriptiven Analysen zum SCM 4.0 Datensatz zugänglich machen. Wenn Sie eine Anleitung wünschen, wählen Sie im untenstehenden Dialo Ja aus. Anschliessend finden Sie einige Informationen zum Projekt im Rahmen der Bachelorarbeit.")
    anleitung_ja = st.radio("Wollen Sie die Anleitung für die Benutzung der Website anzeigen?",("Nein", "Ja"))
    if anleitung_ja == "Ja":
        anleitung()
    st.write(""" 
    #
    #
    Aufbauend auf einem abgeschlossenen Projekt am Institut für Supply Chain Management der Universität St. Gallen ist eine Datenbank ausgewertet worden, die Use Cases zu 4.0 Technologien entlang der Supply Chain enthält. Diese Datenbank wurde aufbereitet und ist die Grundlage für die deskriptiven und präskriptiven Analysen.
    Um den Datensatz und das gesamte Projekt besser zu verstehen, macht es Sinn sich zu Beginn mit den deskriptiven Analysen auseinander zu setzen. Anschliessend kann man auf der Seite der präskriptiven Analysen eine Problemstellung eingeben und erhält eine Technologie vorgeschlagen. 
    Will man dann mehr zu den Use Cases dieser Technologie wissen, kann man zurück auf die Seite der deskriptiven Analysen und dort die Beispiele herausfilten, die diese Technologie enthalten. 
    #
    """)

    img = Image.open(requests.get("https://raw.githubusercontent.com/Bubami/ISCM/main/ISCMHSG_450px.jpg", stream=True).raw)
    img_fhnw = Image.open(requests.get("https://raw.githubusercontent.com/Bubami/ISCM/main/fhnw_450px.jpg", stream=True).raw)
    img_white = Image.open(requests.get("https://raw.githubusercontent.com/Bubami/ISCM/main/white.png", stream=True).raw)

    st.image([np.array(img),np.array(img_white), np.array(img_fhnw)])



    st.write("""
    #### Informationen zum Projekt & Datenschutz
    Dieses Projekt wurde im Rahmen der Bachelorarbeit von Michael Buchbauer am Institut für Supply Chain Management der Universität St. Gallen (ISCM-HSG) erstellt. Die vorgelegte Arbeit basiert auf internen, vertraulichen Daten und Informationen des ISCM. Das herunterladen und weiterverwenden der verwendeten Daten ist nur mit schriftlicher Erlaubnis des ISCM zulässig.
    Die Bachelorarbeit wurde unter folgendem Titel eingereicht: "Konzeption eines prototypischen Recommender Systems basierend auf maschinellem Lernen, zur Empfehlung des Einsatzes von Industrie 4.0 Technologien in der Supply Chain".
    #
    Wir freuen uns über Ihre Rückmeldung auf michael@buchbauer.ch oder stefan.selensky@unisg.ch

    """)

# Deskriptive Analysen
def page_first():
    markdown_breit()

    st.title("""
    Deskriptive Analysen vom Datensatz zum Einsatz von 4.0 Technologien in der Supply Chain
    ---
    """)

    st.header("Datensatz")
    if st.checkbox("Datensatz anzeigen"):

        st.write("""
        ### Der Datensatz wird zuerst statisch abgebildet und beschrieben. Weiter unten können die gewünschten Daten gefiltert werden.
        """)
        st.write("""
        Abbildung des Datensatzes
        """)
        if st.checkbox("Instruktionen anzeigen"):
            st.info("Die Tabelle wird auf-, bzw. absteigend nach dem Attribut geordnet das in der grauen Kopfzeile angewählt wird.")

        st.write(df)
        st.write("")
        st.write("")
        st.write("Beschreibung des Datensatzes")


        if st.checkbox("Zusätzliche Informationen zur Beschreibung anzeigen lassen."):
            st.info("""
                ### Legende
                * count = Anzahl Einträge
                * unique = Anzahl unterschiedlicher Einträge
                * top = Häufigster Eintrag
                * freq = Anzahl Einträge des häufigsten Eintrags
                    """)
        st.write(df.astype(str).describe(include='all'))



    st.header("Filtern & Visualisieren")
    #if st.checkbox("Tabelle filtern"):
    st.write("""
    #### Technologien
    """)
    new_df = pd.DataFrame()
    new_df1 = pd.DataFrame()
    new_df2 = pd.DataFrame()
    auswahl = st.selectbox("Was soll der erste Filter sein?", ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
    tech = st.multiselect('Attribut auswählen', df[auswahl].unique().tolist())
    new_df = df[(df[auswahl].isin(tech))]
    if st.checkbox("Treffer anzeigen"):
        st.write(new_df)
    st.write(new_df.astype(str).describe(include='all'))


    if st.checkbox("Weitere Filter einblenden"):
        st.write("""
        #### Zweiter Filter
        """)
        auswahl1 = st.selectbox("Was soll der zweite Filter sein?", ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
        if auswahl1 == auswahl:
                st.write("Sie haben den gleichen Filter wie oben gewählt")
        else:
                zwei = st.multiselect("Vorhandene Attribute des zweiten Filters", new_df[auswahl1].unique().tolist())
                new_df1 =  new_df[(new_df[auswahl1].isin(zwei))]
                st.write(new_df1)
                st.write(new_df1.astype(str).describe(include='all'))



        auswahl2 = st.selectbox("Was soll der dritte Filter sein?", ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
        if auswahl2 == auswahl1 or auswahl2 == auswahl:
                st.write("Sie haben den gleichen Filter wie oben gewählt")
        else:
                drei = st.multiselect("Vorhandene Attribute des dritten Filters", new_df1[auswahl2].unique().tolist())
                new_df2 =  new_df1[(new_df1[auswahl2].isin(drei))]
                st.write(new_df2)
                st.write(new_df2.astype(str).describe(include='all'))



    if st.checkbox("Einzelne Beispiele detailiert anzeigen"):
        bsp = st.multiselect("Geben Sie die ID Nummer des gewünschten Beispiels ein", df["ID"].unique())
        new_df_all = df_all[(df_all["ID"].isin(bsp))]
        if st.checkbox("Anzeigen"):
            st.write(new_df_all)


    st.header("Visualisieren der Daten")
    selection_list = []
    selection_list.append([])




    st.write("""
    #### Kuchendiagramm
    """)
    if new_df1.empty and new_df2.empty and new_df.empty:
        selection = st.selectbox("Wählen Sie das Attribut für das Kuchendiagramm",  ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
        fig = px.pie(df, names=selection, title='Auteilung der Einträge')
        st.plotly_chart(fig)
        if st.checkbox("2tes Kuchendiagramm anzeigen"):
            selection1 = st.selectbox("Wählen Sie das Attribut für das 2. Kuchendiagramm",  ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
            fig2 = px.pie(df, names=selection1, title='Auteilung der Funktionen')
            st.plotly_chart(fig2)

        if st.checkbox("PPS-Diagramm ausgeben? (Berechnung dauert einen Augenblick)"):
            plt.rcParams['font.size'] = 18.0
            plt.rcParams["figure.figsize"] = (30,20)
            matrix_df = pps.matrix(df).pivot(columns='x',
                                             index='y', values='ppscore')
            sns.heatmap(matrix_df, annot=True)
            st.pyplot()

    elif new_df1.empty and new_df2.empty:
        selection = st.selectbox("Wählen Sie das Attribut für das Kuchendiagramm",  ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
        selection_list[0] = selection
        fig = px.pie(new_df, names=selection, title='Auteilung der Einträge')
        st.plotly_chart(fig)
        if st.checkbox("2tes Kuchendiagramm anzeigen"):
            selection1 = st.selectbox("Wählen Sie das Attribut für das 2. Kuchendiagramm",  ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
            fig2 = px.pie(new_df, names=selection1, title='Auteilung der Funktionen')
            st.plotly_chart(fig2)

        if st.checkbox("PPS-Diagramm ausgeben? (Berechnung dauert einen Augenblick"):
            plt.rcParams['font.size'] = 18.0
            plt.rcParams["figure.figsize"] = (30,20)
            matrix_df = pps.matrix(new_df).pivot(columns='x',
                                             index='y', values='ppscore')
            sns.heatmap(matrix_df, annot=True)
            st.pyplot()

    elif new_df2.empty:
        selection = st.selectbox("Wählen Sie das Attribut für das Kuchendiagramm",  ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
        fig = px.pie(new_df1, names=selection, title='Auteilung der Funktionen')
        st.plotly_chart(fig)

        if st.checkbox("2tes Kuchendiagramm anzeigen"):
            selection1 = st.selectbox("Wählen Sie das Attribut für das 2. Kuchendiagramm",  ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
            fig2 = px.pie(new_df1, names=selection1, title='Auteilung der Funktionen')
            st.plotly_chart(fig2)


        if st.checkbox("PPS-Diagramm ausgeben? (Berechnung dauert einen Augenblick"):
            plt.rcParams['font.size'] = 18.0
            plt.rcParams["figure.figsize"] = (30,20)
            matrix_df = pps.matrix(new_df1).pivot(columns='x',
                                             index='y', values='ppscore')
            sns.heatmap(matrix_df, annot=True)
            st.pyplot()

    else:
        selection = st.selectbox("Wählen Sie das Attribut für das Kuchendiagramm",  ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
        fig = px.pie(new_df2, names=selection, title='Auteilung der Funktionen')
        st.plotly_chart(fig)

        if st.checkbox("2tes Kuchendiagramm anzeigen"):
            selection1 = st.selectbox("Wählen Sie das Attribut für das 2. Kuchendiagramm",  ("Technologie","IoT", "CPS", "CC", "ML", "DLT", "MBS", "Maturity_Index", "Strategische_Ziele", "Branche", "Funktion", "SCOR1" , "Unterprozess" ))
            fig2 = px.pie(new_df2, names=selection1, title='Auteilung der Funktionen')
            st.plotly_chart(fig2)


        if st.checkbox("PPS-Diagramm ausgeben? (Berechnung dauert einen Augenblick)"):
            plt.rcParams['font.size'] = 18.0
            plt.rcParams["figure.figsize"] = (30,20)
            matrix_df = pps.matrix(new_df2).pivot(columns='x',
                                             index='y', values='ppscore')
            sns.heatmap(matrix_df, annot=True)
            st.pyplot()




# Präskriptive Analysen
def page_second():
    markdown_schmal()


    #Einleitung
    st.title("Vorhersagen zum Einsatz von 4.0 Technologien in der Supply Chain")
    st.header("Problemstellung"
              "")
    st.write("Sie können nun Ihre Problemstellung eingeben")
    st.write("Wählen Sie dafür die passendste Eigenschaft jedes Attributs aus. Um Zusatzinformationen einzublenden, wählen Sie die jeweilige Option Ja")


################################################################3
    #Eingabe der Problemstellung


    eingabe()
    st.header("""
    Antworten überprüfen
    """)
    if st.checkbox("Antworten einblenden"):
        st.write("Maturity Index = " + X_test1[0])
        st.write("Strategisches Ziel = " + X_test1[1])
        st.write("Branche = " + X_test1[2])
        st.write("Funktion = " + X_test1[3])
        st.write("SCOR1 = " + X_test1[4])
        st.write("Unterprozess = " + X_test1[5])

    ###############################################################################################
    st.header("""
    Algorithmus
    """)

    st.write("""
    ### Mit welchem Algorithmus soll die Vorhersage gemacht werden?
    """)
    display = ("KNN", "Naiver Bayes", "Random Forest")
    options = list(range(len(display)))
    value = st.selectbox("Wählen sie den gewünschten Algorithmus", options, format_func=lambda x: display[x])

    #Anzeige der Erklärung zu den Algorithmen
    if st.checkbox("Zusatzinformationen zu den Algorithmen anzeigen"):
        algorithmen_info()


    #Eingabe der n Nachbarn für den KNN Algorithmus
    if value == 0:
        nachbarn = st.slider("Bestimmen Sie das K für den KNN Algorithmus",1,50,20)

    #Eingabe der n Estimatoren für den RandomForest Algorithmus
    if value == 2:
        n_estimatoren = st.slider("Bestimmen sie die n Estimatoren für den RandomForest Algorithmus",5,80,30)



    #Eingabe der Testgrösse
    testgroesse = st.slider("Wählen Sie die Testgrösse (Massgebend für die Aufteilung der Trainings-, und Testdaten)",0.01,0.9,0.15)
    #testgroesse = 0.15
    #Eingabe der Zufallszahl (Randomstate)
    #zufallszahl = st.slider("Wählen Sie den Randomstate (Massgebend für Durchmischung der Trainings-, und Testdaten)",1,20,1)
    zufallszahl = 1





    df_without_tech = df
    df_without_tech = df_without_tech.drop("Technologie", axis=1)
    df_without_tech = df_without_tech.drop("IoT", axis=1)
    df_without_tech = df_without_tech.drop("CPS", axis=1)
    df_without_tech = df_without_tech.drop("CC", axis=1)
    df_without_tech = df_without_tech.drop("ML", axis=1)
    df_without_tech = df_without_tech.drop("DLT", axis=1)
    df_without_tech = df_without_tech.drop("MBS", axis=1)
    df_without_tech = df_without_tech.drop('ID', axis=1)
    modDf = df_without_tech.append(pd.Series(X_test1, index=df_without_tech.columns), ignore_index=True)

    modDf.astype(str).describe(include='all')
    # Import LabelEncoder
    from sklearn import preprocessing
    # creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    Mi = le.fit_transform(modDf["Maturity_Index"].values)
    SZ = le.fit_transform(modDf["Strategische_Ziele"].values)
    Br = le.fit_transform(modDf["Branche"].values)
    Fkt = le.fit_transform(modDf["Funktion"].values)
    SCR = le.fit_transform(modDf["SCOR1"].values)
    UP = le.fit_transform(modDf["Unterprozess"].values)

    features = list(zip(Mi, SZ, Br, Fkt, SCR, UP))  # ,luftfeuchtigkeit_encoded,wind_encoded))
    X = features

    Tech = le.fit_transform(df["Technologie"].values)
    IoT = le.fit_transform(df["IoT"].values)
    CPS = le.fit_transform(df["CPS"].values)
    CC = le.fit_transform(df["CC"].values)
    ML = le.fit_transform(df["ML"].values)
    DLT = le.fit_transform(df["DLT"].values)
    MBS = le.fit_transform(df["MBS"].values)

    y = Tech
    y_IoT = IoT
    y_CPS = CPS
    y_CC = CC
    y_ML = ML
    y_DLT = DLT
    y_MBS = MBS

    # Der erstellte und nun kodierte Datensatz wird in eine Variable gespeichert und wieder vom grossen Datensatz gelöscht
    laenge = len(X)
    X_test1_kodiert = [X[laenge-1]]
    del X[laenge-1]

    testsize = testgroesse
    randomstate = zufallszahl
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=randomstate, test_size=testsize)
    X_train, X_test, y_IoT_train, y_IoT_test = train_test_split(X, y_IoT, random_state=randomstate, test_size=testsize)
    X_train, X_test, y_CPS_train, y_CPS_test = train_test_split(X, y_CPS, random_state=randomstate, test_size=testsize)
    X_train, X_test, y_CC_train, y_CC_test = train_test_split(X, y_CC, random_state=randomstate, test_size=testsize)
    X_train, X_test, y_ML_train, y_ML_test = train_test_split(X, y_ML, random_state=randomstate, test_size=testsize)
    X_train, X_test, y_DLT_train, y_DLT_test = train_test_split(X, y_DLT, random_state=randomstate, test_size=testsize)
    X_train, X_test, y_MBS_train, y_MBS_test = train_test_split(X, y_MBS, random_state=randomstate, test_size=testsize)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    X_eingegeben = scaler.transform(X_test1_kodiert)
    ##################################################################################

    ##################################################################################
    if value == 0:
        params = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,31,32,33,34,35,36,37,38,39,40,41,41,43,44,45,46,47,48,49,50,51,52,52,53,54,55,56,57,58,59,60]}

        knn = neighbors.KNeighborsRegressor()

        model1 = GridSearchCV(knn, params, cv=10)
        model1.fit(X_train_transformed,y_train)
        k_value = model1.best_params_

        model = KNeighborsClassifier(n_neighbors=nachbarn)
        model.fit(X_train_transformed, y_train)
        y_pred = model.predict(X_eingegeben)
        probar = (model.predict_proba(X_eingegeben))
        probar.sort()
        modelscore_Technologie = model.score(X_test_transformed, y_test)
        y = le.fit_transform(df["Technologie"].values)
        Technologie = le.inverse_transform(y_pred)
        np.array(Technologie).tolist()



        model.fit(X_train_transformed, y_IoT_train)
        modelscore_IoT = model.score(X_test_transformed, y_IoT_test)
        y_IoT_pred = model.predict(X_eingegeben)
        y_IoT = le.fit_transform(df["IoT"].values)
        IoT = le.inverse_transform(y_IoT_pred)
        np.array(IoT).tolist()

        model.fit(X_train_transformed, y_CPS_train)
        modelscore_CPS = model.score(X_test_transformed, y_CPS_test)
        y_CPS_pred = model.predict(X_eingegeben)
        y_CPS = le.fit_transform(df["CPS"].values)
        CPS = le.inverse_transform(y_CPS_pred)
        np.array(CPS).tolist()

        model.fit(X_train_transformed, y_CC_train)
        modelscore_CC = model.score(X_test_transformed, y_CC_test)
        y_CC_pred = model.predict(X_eingegeben)
        y_CC = le.fit_transform(df["CC"].values)
        CC = le.inverse_transform(y_CC_pred)
        np.array(CC).tolist()

        model.fit(X_train_transformed, y_ML_train)
        modelscore_ML = model.score(X_test_transformed, y_ML_test)
        y_ML_pred = model.predict(X_eingegeben)
        y_ML = le.fit_transform(df["ML"].values)
        ML = le.inverse_transform(y_ML_pred)
        np.array(ML).tolist()

        model.fit(X_train_transformed, y_DLT_train)
        modelscore_DLT = model.score(X_test_transformed, y_DLT_test)
        y_DLT_pred = model.predict(X_eingegeben)
        y_DLT = le.fit_transform(df["DLT"].values)
        DLT = le.inverse_transform(y_DLT_pred)
        np.array(DLT).tolist()

        model.fit(X_train_transformed, y_MBS_train)
        modelscore_MBS = model.score(X_test_transformed, y_MBS_test)
        y_MBS_pred = model.predict(X_eingegeben)
        y_MBS = le.fit_transform(df["MBS"].values)
        MBS = le.inverse_transform(y_MBS_pred)
        np.array(MBS).tolist()
    ####################################################################################
    # Naive Bayes Algorithmus
    if value == 1:
        model = GaussianNB()
        model.fit(X_train_transformed, y_train)
        print(model.score(X_test_transformed, y_test))

        y_pred = model.predict(X_eingegeben)
        modelscore_Technologie = model.score(X_test_transformed, y_test)
        y = le.fit_transform(df["Technologie"].values)
        Technologie = le.inverse_transform(y_pred)
        np.array(Technologie).tolist()

        model.fit(X_train_transformed, y_IoT_train)
        modelscore_IoT = model.score(X_test_transformed, y_IoT_test)
        y_IoT_pred = model.predict(X_eingegeben)
        y_IoT = le.fit_transform(df["IoT"].values)
        IoT = le.inverse_transform(y_IoT_pred)
        np.array(IoT).tolist()

        model.fit(X_train_transformed, y_CPS_train)
        modelscore_CPS = model.score(X_test_transformed, y_CPS_test)
        y_CPS_pred = model.predict(X_eingegeben)
        y_CPS = le.fit_transform(df["CPS"].values)
        CPS = le.inverse_transform(y_CPS_pred)
        np.array(CPS).tolist()

        model.fit(X_train_transformed, y_CC_train)
        modelscore_CC = model.score(X_test_transformed, y_CC_test)
        y_CC_pred = model.predict(X_eingegeben)
        y_CC = le.fit_transform(df["CC"].values)
        CC = le.inverse_transform(y_CC_pred)
        np.array(CC).tolist()

        model.fit(X_train_transformed, y_ML_train)
        modelscore_ML = model.score(X_test_transformed, y_ML_test)
        y_ML_pred = model.predict(X_eingegeben)
        y_ML = le.fit_transform(df["ML"].values)
        ML = le.inverse_transform(y_ML_pred)
        np.array(ML).tolist()

        model.fit(X_train_transformed, y_DLT_train)
        modelscore_DLT = model.score(X_test_transformed, y_DLT_test)
        y_DLT_pred = model.predict(X_eingegeben)
        y_DLT = le.fit_transform(df["DLT"].values)
        DLT = le.inverse_transform(y_DLT_pred)
        np.array(DLT).tolist()

        model.fit(X_train_transformed, y_MBS_train)
        modelscore_MBS = model.score(X_test_transformed, y_MBS_test)
        y_MBS_pred = model.predict(X_eingegeben)
        y_MBS = le.fit_transform(df["MBS"].values)
        MBS = le.inverse_transform(y_MBS_pred)
        np.array(MBS).tolist()

    ####################################################################################
    # Random Forest Algorithmus
    if value == 2:
        model = RandomForestClassifier(criterion="entropy", n_estimators=n_estimatoren)
        model.fit(X_train_transformed, y_train)
        print(model.score(X_test_transformed, y_test))

        y_pred = model.predict(X_eingegeben)
        modelscore_Technologie = model.score(X_test_transformed, y_test)
        y = le.fit_transform(df["Technologie"].values)
        Technologie = le.inverse_transform(y_pred)
        np.array(Technologie).tolist()

        model.fit(X_train_transformed, y_IoT_train)
        modelscore_IoT = model.score(X_test_transformed, y_IoT_test)
        y_IoT_pred = model.predict(X_eingegeben)
        y_IoT = le.fit_transform(df["IoT"].values)
        IoT = le.inverse_transform(y_IoT_pred)
        np.array(IoT).tolist()

        model.fit(X_train_transformed, y_CPS_train)
        modelscore_CPS = model.score(X_test_transformed, y_CPS_test)
        y_CPS_pred = model.predict(X_eingegeben)
        y_CPS = le.fit_transform(df["CPS"].values)
        CPS = le.inverse_transform(y_CPS_pred)
        np.array(CPS).tolist()

        model.fit(X_train_transformed, y_CC_train)
        modelscore_CC = model.score(X_test_transformed, y_CC_test)
        y_CC_pred = model.predict(X_eingegeben)
        y_CC = le.fit_transform(df["CC"].values)
        CC = le.inverse_transform(y_CC_pred)
        np.array(CC).tolist()

        model.fit(X_train_transformed, y_ML_train)
        modelscore_ML = model.score(X_test_transformed, y_ML_test)
        y_ML_pred = model.predict(X_eingegeben)
        y_ML = le.fit_transform(df["ML"].values)
        ML = le.inverse_transform(y_ML_pred)
        np.array(ML).tolist()

        model.fit(X_train_transformed, y_DLT_train)
        modelscore_DLT = model.score(X_test_transformed, y_DLT_test)
        y_DLT_pred = model.predict(X_eingegeben)
        y_DLT = le.fit_transform(df["DLT"].values)
        DLT = le.inverse_transform(y_DLT_pred)
        np.array(DLT).tolist()

        model.fit(X_train_transformed, y_MBS_train)
        modelscore_MBS = model.score(X_test_transformed, y_MBS_test)
        y_MBS_pred = model.predict(X_eingegeben)
        y_MBS = le.fit_transform(df["MBS"].values)
        MBS = le.inverse_transform(y_MBS_pred)
        np.array(MBS).tolist()

    ####################################################################################3
    st.header("""
    Empfehlung
    """)

    if value == 0:
        st.write("Das empfohlene K für KNN ist:")
        st.write(k_value)
    st.write("""
    ### Die für das eingegebene Problem vorgeschlagene Technologie ist: 
    """)
    if X_test1[0] != "ALL" or X_test1[1] != "ALL" or X_test1[2] != "ALL" or X_test1[3] != "ALL" or X_test1[
        4] != "ALL" or X_test1[5] != "ALL":
        st.success(Technologie[0])
        st.write(probar)
        st.write("""
        Bei einer Genauigkeit des Modells von: 
        """)
        st.write(modelscore_Technologie)
    else:
        st.error("Geben Sie eine Problemstellung ein")


    st.write("""
    ### Das Modell errechnet für folgende Basis-Technologien, dass deren Einsatz sinnvoll sein könnte:
    """)
    if X_test1[0] != "ALL" or X_test1[1] != "ALL" or X_test1[2] != "ALL" or X_test1[3] != "ALL" or X_test1[
        4] != "ALL" or X_test1[5] != "ALL":
        if IoT == ['WAHR']:
            if modelscore_IoT > 0.7:
                st.write('- IoT')
        if CPS == ['WAHR']:
            if modelscore_CPS > 0.7:
                st.write('- Cyber Physical Systems')
        if CC == ['WAHR']:
            if modelscore_CC > 0.7:
                st.write('- Cloud Computing')
        if MBS == ['WAHR']:
            if modelscore_ML > 0.7:
                st.write('- Machine Learning')
        if DLT == ['WAHR']:
            if modelscore_DLT > 0.7:
                st.write('- Distributed Ledger Technology')
        if MBS == ['WAHR']:
            if modelscore_MBS > 0.7:
                st.write('- Mobile Based Systems')


    else:
        st.error("Geben Sie eine Problemstellung ein")


    st.header("Bewertung")
    beschreibung = st.text_input("Wir bitten Sie das eingegebene Technologieprojekt in 2-3 Sätzen zu beschreiben: ")
    if st.checkbox("Beispiel anzeigen)"):
        st.write("Bsp.: China ist auf der Überholspur zu Drohnenlieferungen. JD.com und die SF Holding bauen Netzwerke von großen und kleinen UAVs auf und arbeiten mit den Regulierungsbehörden zusammen, um Regeln für einen breiten Einsatz zu schaffen. Die Idee besteht darin, ein Netzwerk aufzubauen, das nicht nur kleine Drohnen für die Endauslieferung umfasst, sondern ein ganzes System, einschließlich großer autonomer Starrflügelflugzeuge, die von kleinen Flughäfen oder Landebahnen starten, um Massengüter zwischen den Lagerhäusern zu befördern. JD strebt an, die kleinen Last-hop-Flüge mit größeren Drohnen zu integrieren, die Waren an Verteilungszentren und traditionelle Frachtflugzeuge weiterleiten, um das Ziel zu erreichen, dass alle Lieferungen innerhalb Chinas in weniger als 36 Stunden abgeschlossen sind.")
    bewertung = st.slider("Wie zufrieden sind Sie mit der Empfehlung der Technologie?",1,10)
    ideale_technologie = ''
    quelle = ''
    if st.checkbox("Ich halte eine andere Technologie für sinnvoller"):
        technologie_eingabe = st.radio("Möchten Sie die Technologie selber eingeben oder aus dem Dropdown auswählen?",("Eingabe", "Dropdown"))
        if technologie_eingabe == "Dropdown":
            ideale_technologie = st.selectbox("Welche Technologie halten Sie für angebrachter?",unique_sorted_values_plus_ALL(df.Technologie))
        else:
            ideale_technologie = st.text_input("Geben Sie ihren Technologievorschlag ein:")
    if st.checkbox("Quelle zum eingegebenen Problem erfassen"):
        quelle = st.text_input("Geben Sie die URL der Quelle ein.")

    Attribute = 22
    l_bewertung = []
    for i in range(Attribute):
        l_bewertung.append([])
    for a in range(0,6):
        l_bewertung[a] = X_test1[a]
    #Parameter für Algorithmen (KNN Nachbarn und Random Forest n Estimatoren
    if value == 0:
        l_bewertung[6] = "KNN"
        l_bewertung[7] = nachbarn
    elif value == 2:
        l_bewertung[7] = n_estimatoren
        l_bewertung[6] = "RandomForest"
    else:
        l_bewertung[6] = "Naive Bayes"
        l_bewertung[7] = "x"
    l_bewertung[8] = testgroesse
    l_bewertung[9] = zufallszahl
    l_bewertung[10] = Technologie[0]
    l_bewertung[11] = modelscore_Technologie
    l_bewertung[12] = modelscore_IoT
    l_bewertung[13] = modelscore_CPS
    l_bewertung[14] = modelscore_CC
    l_bewertung[15] = modelscore_ML
    l_bewertung[16] = modelscore_DLT
    l_bewertung[17] = modelscore_MBS
    l_bewertung[18] = bewertung
    l_bewertung[19] = ideale_technologie
    l_bewertung[20] = beschreibung
    l_bewertung[21] = quelle



    #df_ausgabe =
    if st.button('Bewertung abschliessen'):
        st.write("""
        Factsheet
        """)
        df_ausgabe = DataFrame(l_bewertung)#,
               #columns=['Maturity_Index', 'Strategische_Ziele',
                    #    'Branche', 'Funktion', 'SCOR1', 'Unterprozess','Algorithmus', 'N-Nachbarn (KNN)' ,'N-Estimatoren (RandomForest)','Testgroesse','Random state','Technologie','IoT', 'CPS', 'CC', 'ML', 'DLT', 'MBS', 'Bewertung'])
        st.write(df_ausgabe)
        csv = df_ausgabe.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown('### **⬇️ Download output CSV File **')
        href = f'<a href="data:file/csv;base64,{b64}" download="Factsheet.csv">Download csv file</a>'

        st.markdown(href, unsafe_allow_html=True)
    st.write("""
    Wir bitten Sie das Factsheet.csv zur Verbesserung der Vorhersagegenauigkeit an stefan.selensky@unisg.ch zu senden.
    """)




#Informationsblöcke mit Beschreibung für Streamlit
def anleitung():
    st.info("""
    ### Startseite (Allgemeine Informationen, Anleitung)
     * Mit der Auswahlleiste links, kann zwischen deskriptiv und präskriptiv gewählt werden. Ist keine Leiste sichtbar muss oben links auf das Pfeilsymbol gedrückt werden.
    ### Deskriptiv (Datensatz, Numerische Informationen, Deskriptive Analysen)
     * Auf der Seite für die deskriptiven Analysen, ist der Datensatz 1:1 abgebildet und die Eckdaten dazu aufgelistet. 
     * Man hat die Möglichkeit die Daten nach einzelnen Parametern zu filtern um den Datensatz gefilterten anzuschauen. Bis zu drei Filterstufen sind möglich.
     * Gewünschte Beispiele können mit Beschreibung und Quelle ausgegeben werden
     * Die gefilterten Daten können weiter unten nochmals sortiert in einem Kuchendiagramm dargestellt werden, oder es kann ein Predictive-Power-Score Diagramm gezeichnet werden.
     
       
    ### Präskriptiv (Eingabe der Problemstellung, Vorhersage, Bewertung der Vorhersage)
     * Auf der Seite für die präskriptiven Analyen ist eine Eingabe der Problemparater eingerichtet.
     * Am Ende wird eine Technologie, sowie passende Schlüsseltechnologien vorgeschlagen.
     * Zusätzliche Informationen können falls vorhanden mit einem Button ein-, bzw. ausgeblendet werden.
     * Im untersten Teil kann die Empfehlung bewertet werden.
     * Die .csv Datei mit Ihrer Bewertung soll ans ISCM-HSG gesendet werden.

    """)
def maturity_info():
    st.info("""
        #### Computerisierung 
        * Ausgangspunkt und Grundlage für Digitalisierung und oft schon weit vorangeschritten bei den Unternehmen
        * Isolierter Einsatz von IT (Informationstechnologien)
        * Bei der Bewältigung von repetitiven Tätigkeiten
        
        #### Konnektivität
        * Vernetzung der isoliert eingesetzten IT 
        * Es gibt noch keine vollständige Integration von Operativen Technologien mit der IT
        
        #### Sichtbarkeit
        * Sensoren erfassen Prozesse von Anfang bis Ende mit einer Vielzahl von Datenpunkten
        * Digitales Abbild des Unternehmens kann erzeugt werden
        
        #### Transparenz
        * Das Unternehmen versteht durch Ursachenanalysen, warum etwas passiert, und leiten daraus Wissen über Wirkungszusammenhänge ab
        
        
        #### Prognosefähigkeit
        * Zukunftsszenarien lassen sich simulieren und die wahrscheinlichsten lassen sich identifizieren
        * Maßnahmen müssen in der Regel noch manuell eingeleitet werden, durch die gewonnene Vorwarnzeit können die Auswirkungen der Störung jedoch frühzeitig begrenzt werden
    
        
        #### Adaptierbarkeit
        * Automatisches Handeln und die Selbstoptimierung
        * Entscheidungen werden IT-Systemen überlassen
        * Daten des digitalen Schattens werden so eingesetzt, dass Entscheidungen mit den größten positiven Auswirkungen autonom und ohne menschliches Zutun in kürzester Zeit getroffen und die daraus resultierenden Maßnahmen umgesetzt werden
        """
        )
def funku_info():
    st.info("""
        #### Analyse
        * Zerlegung von Daten/ Informationen in ihre Bestandteile und Neukombination dieser mit dem Ziel Erkenntnisse ableiten zu können
        #### Datenerfassung
        * Aufnahme und evtl. Umwandlung bestehnder digitaler Daten oder Messung von Eigenschaften aus der Umwelt durch Sensoren und Umwandlung zu verwertbaren digitalen Daten  
        #### Kommunikation
        * Übertragung von Nachrichten zwischen einem Sender und einem oder mehreren Empfängern.

        #### Örtliche Sendungsverfolgung und Rückverfolgung
        * Sendungsverfolgung = Auskunft darüber wo sich Waren zu einem bestimmten Zeitpunkt befinden 
        * Rückverfolgung = nachträgliche Auskunft über den gesamten Versand- und Herstellungsprozess
                
        #### Planung 
        * Erstellung eines organisierten Vorschlags für zukünftiges Vorgehen anhand gegebener Informationen
        * Mögliche Formen können festgelegte Aktionen, ein Programm oder Strategien sein.
        
        #### Prognose
        * Beschreibung über mögliche Ereignisse oder Entwicklungen in der Zukunft anhand gegebener Informationen
        
        #### Speicherung und Bereitstellung 
        * Speicherung, Organisation und Wartung von Daten sowie Ermöglichen des möglichst einfachen Zugriffs. Insbesondere Aufbau und Pflege von Datenbanken
        #### Transaktion
        * Vertragliche Vereinbarung des Austausch von SC-Flüssen (Güter-, Informations- Rechts- und Geldfüssen)
        #### Transport
        * Überbringen von Gütern von einem Ort zum anderen. Hierzu zählt der inner- und ausserbetriebliche Transport.
        #### Unterstützung der MA
        * Bisherige von Mitarbeiter ausgeführte Prozesse werden durch Erweiterung deren Kapazitäten und Fähigkeiten optimiert. Der Mitarbeiter wird nicht ersetzt.
        #### Verifizierung
        * Bestätigung, dass festgelegte Forderungen erfüllt werden / worden sind        
        #### Zustandsbestimmung
        * Zusammenführung und Darstellung relevanter Zustandsdaten eines Objekts. Darunter fällt beispielsweise die Erkennung von Veränderungen in Leistung, Qualität, Auslastung und Verschleiss und Bestimmung der Ursache. 

        

        

        """
        )
def scor_info():
    st.info("""
    #### Plan
    * Planung der Supply Chain Steuerung
    * Sammeln von Vorschriften und anderen Voraussetzungen 
    * Sammeln von Informationen über verfügbare Ressourcen
    * Ausbalancieren von Vorschriften und Ressourcen um die Verfügbaren Kapazitäten und möglichen Lücken zu identifizieren 
    * Aktionspläne gegen potenzielle Lücken entwerfen
    
    #### Source
    * Bestellauftrag, Entgegennahme, Validierung und Lagerung von Gütern und Dienstleistungen
    * Entgegennahme/ Bearbeitung der Lieferantenrechnung
    
    #### Make
    * Verarbeitung von Materialien oder Erstellung von Dienstleistungen, darunter fällt die Fertigung, die Chemische Bearbeitung, die Wartung, die Reparatur, die Überarbeitung, das Recycling, die Sanierung und andere Bereiche der Wandlung von Gütern. 
    * Faustregel: Ein oder mehr Güter werden verwendet und ein oder mehr andere Güter entstehen
    
    #### Deliver
    * Entgegennahme, Pflege und Erfüllung von Kundenaufträgen
    * Die Planung der Lieferung
    * Verpackung und Rechnungsstellung
    
    #### Return
    * Den rückläufigen Güterfluss
    * Identifikation der Rücksendungsnotwendigkeit
    * Planung der Rücksendung
    * Versand und Entgegennahme retournierter Güter
    * Achtung: Reparatur, Recycling und Sanierung sind Teil des Make Schrittes!
        
    #### Enable
    * Das generelle Management der Supply Chain 
    * Management von Geschäftsregeln, Performance Management, Ressourcen Management, Facility Management, Vertrags Management, Daten Management, Supply Chain Netzwerk Management, Compliance und Risk Management
    
    """)
def up_info():
    st.info("""
    #### Plan
    * Plan1-2 = Identify, Prioritize and Aggregate SC Requirements 
    * Plan3 = Balance SC Resources with SC Requirements
    * Plan4 = Establish and Communicate SC Plans
    #### Source
    * Soruce1-3 = Identify Sources of Supply; Select Final Supplier and Negotiate; Schedule Product Deliveries
    * Source4-6 = Receive Product; Verify Product; Transfer Product
    * Source7 = Authorize Supplier Payment
    #### Make
    * Make1-2 = Finalize Production Engineering; Schedule Production Activities
    * Make3 = Issue Sourced/-In-Process Product
    * Make4 = Produce and Test
    * Make5 = Package
    * Make6-8 = Stage Finished Product; Release Finished Product to Deliver
    #### Deliver
    * Deliver1-3 = Process Inquiry and Quote; Receive, Enter and Validate Order; Reserve Inventory and Determine Delivery Date
    * Deliver4-7 = Consolidate Orders; Build Loads; Route Shipments; Select Carriers and Rate Shipments
    * Deliver8-11 = Receive Product from Source or Make; Pick Product; Pack Product; Load Product & Generate Shipping Documents
    * Deliver 12 = Ship Product
    * Deliver13-15 = Receive and Verify Product by Costumer; Install Product; Invoice
    #### Return
    * Return1-2 = Identify Product Condition; Disposition Product
    * Return3-4 = Request Product Return Authorization, Schedule Product Shipment
    * Return5-6 = Return Product; Receive Product (includes verify)
    #### Enable
    * Enable1-2 = Manage SC Business rules & Performance
    * Enable3 = Manage Data and Information
    * Enable4-7 = Manage SC Human Ressources, Assets, Contracts & Network
    * Enable8 = Manage Regulatory Compliance
    * Enable9-11 = Manage SC Risk, Procurement & Technology



    """)
def algorithmen_info():
    st.info("""
    #### Allgemein
    Der Autor empfiehlt Naive Bayes als passendster Algorithmus, jedoch auch die anderen können gute Resultate bringen.
    * Testgrösse bestimmt, wieviel der Daten zum Testen des Modells verwendet werden (bspw. 0.1 = 10 % der Daten dienen als Testdaten, der Rest als Trainingsdaten)
      * Hat direkten Einfluss auf die Genauigkeit des Modells   
      * Empfehlung für Testgrösse zwischen  0.1 - 0.3
    * Zufallszahl (Randomstate) bestimmt, wie die Trainings- und Testdaten durchmischt werden. Wird der Randomstate verändert, so werden die Daten neu durchmischt. Bei gleichbleibendem Randomstate, bleibt auch die Durchmischung gleich.
      * Hat keinen direkten Einfluss auf die Genauigkeit des Modells
    
    
    #### Naive Bayes
    Der Naive Bayes Klassifikationsalgorithmus zählt ebenfalls zu den simpel einzusetzenden Algorithmen und basiert auf dem Bayes-Theorem. Es bestimmt, mit welcher bedingten Wahrscheinlichkeit Y eintrifft, wenn X eingetroffen ist. 
    #### Random Forest
    Der Random Forest besteht, wie der Name schon sagt, aus einer grossen Anzahl von einzelnen Entscheidungsbäumen, die als Ganzes funktionieren. Jeder einzelne Baum im Random Forest erzeugt eine Klassenvorhersage und die Klasse mit den meisten Stimmen wird die Vorhersage in dem Modell. 
    Die Anzahl zu erstellender Entscheidungsbäume kann als Parameter in den Algorithmus eingegeben werden. Eine hoher Wert führt jedoch schnell dazu, dass viel Rechenleistung benötigt wird.
    Empfehlung für n Zufallsbäume = 10
    
    #### K-Nearest Neighbour
    Der KNN Klassifikations Algorithmus ist ein einfach zu implementierender Supervised Learning Algorithmus
    KNN geht davon aus, dass ähnliche Dinge in unmittelbarer Nähe existieren. Mit anderen Worten: Ähnliche Dinge sind nahe beieinander. KNN funktioniert, indem es die Abstände zwischen einer Eingabe und allen Beispielen  in den Daten ermittelt, die angegebene Anzahl K von Beispielen auswählt, die der Eingabe am nächsten liegen, und dann, im Falle der Klassifizierung, für das häufigste Etikett stimmt.
    Empfehlung wird K wird unter Empfehlung ausgegeben
    
    """)

def markdown_schmal():
    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {1200}px;
            padding-top: {1}rem;
            padding-right: {3}rem;
            padding-left: {3}rem;
            padding-bottom: {1}rem;
        }}

    </style>
    """,
        unsafe_allow_html=True,
    )
def markdown_breit():
    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {2000}px;
            padding-top: {1}rem;
            padding-right: {3}rem;
            padding-left: {3}rem;
            padding-bottom: {1}rem;
        }}

    </style>
    """,
        unsafe_allow_html=True,
    )



if __name__ == "__main__":
    main()
