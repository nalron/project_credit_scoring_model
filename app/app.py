import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
from zipfile import ZipFile
from sklearn.cluster import KMeans
plt.style.use('fivethirtyeight')
#sns.set_style('darkgrid')


def main() :

    @st.cache
    def load_data():
        z = ZipFile("data/default_risk.zip")
        data = pd.read_csv(z.open('default_risk.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

        z = ZipFile("data/X_sample.zip")
        sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

        target = data.iloc[:, -1:]

        return data, sample, target


    def load_model():
        '''loading the trained model'''
        pickle_in = open('model/LGBMClassifier.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf


    @st.cache(allow_output_mutation=True)
    def load_knn(sample):
        knn = knn_training(sample)
        return knn


    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets


    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

    @st.cache
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]/365), 2)
        return data_age

    @st.cache
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    def load_prediction(sample, id, clf):
        X=sample.iloc[:, :-1]
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        return score

    def load_kmeans(sample, id, mdl):
        index = sample[sample.index == int(id)].index.values
        index = index[0]
        data_client = pd.DataFrame(sample.loc[sample.index, :])
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, data], axis=1)
        return df_neighbors.sample(10)

    def knn_training(sample):
        knn = KMeans(n_clusters=2).fit(sample)
        return knn 



    #Loading data……
    data, sample, target = load_data()
    id_client = sample.index.values
    clf = load_model()


    #######################################
    # SIDEBAR
    #######################################

    #Title display
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
    </div>
    <p style="color: white; font-size: 20px; font-weight: bold; text-align:center">Credit decision support…</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    #Customer ID selection
    st.sidebar.header("**General Info**")

    #Loading selectbox
    chk_id = st.sidebar.selectbox("Client ID", id_client)

    #Loading general info
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)


    ### Display of information in the sidebar ###
    #Number of loans in the sample
    st.sidebar.markdown("<u>Number of loans in the sample :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    #Average income
    st.sidebar.markdown("<u>Average income (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    #AMT CREDIT
    st.sidebar.markdown("<u>Average loan amount (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)
    
    #PieChart
    #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)
    

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    #Display Customer ID from Sidebar
    st.write("Customer ID selection :", chk_id)


    #Customer information display : Customer Gender, Age, Family status, Children, …
    st.header("**Customer information display**")

    if st.checkbox("Show customer information ?"):

        infos_client = identite_client(data, chk_id)
        st.write("Gender :**", infos_client["CODE_GENDER"].values[0], "**")
        st.write("Age :", int(infos_client["DAYS_BIRTH"]/365), "ans")
        st.write("Family status :**", infos_client["NAME_FAMILY_STATUS"].values[0], "**")
        st.write("Number of children :**", infos_client["CNT_CHILDREN"].values[0], "**") 

        #Age distribution plot
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
        ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)
    
        
        st.subheader("*Income (USD)*")
        st.write("Income total :", infos_client["AMT_INCOME_TOTAL"].values[0])
        st.write("Credit amount :", infos_client["AMT_CREDIT"].values[0])
        st.write("Credit annuities :", infos_client["AMT_ANNUITY"].values[0])
        st.write("Amount of property for credit :", infos_client["AMT_GOODS_PRICE"].values[0])
        
        #Income distribution plot
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
        st.pyplot(fig)
        
        #Relationship Age / Income Total interactive plot 
        data_sk = data.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH']/365).round(1)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

        fig.update_layout({'plot_bgcolor':'#f0f0f0', 'paper_bgcolor':'#f0f0f0'}, 
                          title={'text':"Relationship Age / Income Total"}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))

        st.plotly_chart(fig)
    
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    #Customer solvability display
    st.header("**Customer file analysis**")
    st.markdown("<u>Default probability :</u>", unsafe_allow_html=True)
    prediction = load_prediction(sample, chk_id, clf)
    st.write(round(float(prediction)*100, 2), "%")

    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id)) 
    
    #Feature importance
    st.markdown("<u>Feature importance :</u>", unsafe_allow_html=True)
    number = st.slider("Pick a number of features", 0, 50, 10)

    indices = np.argsort(load_model().feature_importances_)[::-1]
    features = []
    for i in range(number):
        features.append(sample.columns[indices[i]]) 

    fig, ax = plt.subplots(figsize=(10, 15))
    sns.barplot(y=features, x=load_model().feature_importances_[indices[range(number)]], 
                color="goldenrod")
    plt.xlabel('')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    #Similar customer files display
    chk_voisins = st.checkbox("Show similar customer files ?")

    if chk_voisins:
        knn = load_knn(sample)
        st.markdown("<u>List of the 10 files closest to this Customer :</u>", unsafe_allow_html=True)
        st.dataframe(load_kmeans(sample, chk_id, knn))
        st.markdown("<i>Target 1 = Customer with default</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()