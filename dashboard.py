"""Dashboard de l'application loan P7"""


import requests
import json
import shap
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from uuid import uuid4


def main():
    """
    C'est la fonction principale qui permet de lancer tout le dashboard.
    """
    # URL local à remplacer une fois l'API déployée sur heroku:
    API_URL = "https://loan-pred-api.herokuapp.com/app"
    # -----------------------------------------------
    # Configuration of the streamlit page
    # -----------------------------------------------
    st.set_page_config(page_title='Dashboard - Application de scoring de crédit',
                       page_icon='🏦',
                       layout='centered',
                       initial_sidebar_state='auto')
    # Display the title
    st.title('Dashboard - Application de scoring de crédits')
    st.subheader("Alexandre Barakat - Data Scientist")

    # Afficher le logo:
    img = Image.open("LOGO.png")
    st.sidebar.image(img, width=250)

    # Afficher l'image d'entête:
    img = Image.open("loan.png")
    st.image(img, width=100)

    # Fonctions:
    def get_list_display_features(f, def_n, key):
        all_feat = f
        n = st.slider("Nombre de variable à afficher",
                      min_value=2, max_value=40,
                      value=def_n, step=None, format=None, key=key)

        disp_cols = list(get_features_importances().sort_values(ascending=False).iloc[:n].index)

        box_cols = st.multiselect(
            'Choisir la variable à afficher:',
            sorted(all_feat),
            default=disp_cols, key=key)
        return box_cols

    ###############################################################################
    #                      LES FONCTIONS D'APPEL A L'API
    ###############################################################################
    # Get list of ID (cached)
    @st.cache(suppress_st_warning=True)
    def get_id_list():
        # URL pour récupérer tous les id d'utilisateurs:
        id_api_url = f"{API_URL}/id"
        # Requesting the API and saving the response
        response = requests.get(id_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        id_customers = pd.Series(content['data']).values
        return id_customers

    # Get selected customer's data (cached)
    data_type = []

    @st.cache
    def get_selected_cust_data(selected_id):
        # URL of the sk_id API
        data_api_url = f"{API_URL}/data_cust/?SK_ID_CURR={str(selected_id)}"
        # Requesting the API and saving the response
        response = requests.get(data_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        x_custom = pd.DataFrame(content['data'])
        # x_cust = json_normalize(content['data'])
        y_customer = (pd.Series(content['y_cust']).rename('TARGET'))
        # y_customer = json_normalize(content['y_cust'].rename('TARGET'))
        return x_custom, y_customer

    # Get score (cached)
    @st.cache
    def get_score_model(selected_id):
        # URL of the sk_id API
        score_api_url = f"{API_URL}/customer_score/?SK_ID_CURR={str(selected_id)}"
        # Requesting the API and saving the response
        response = requests.get(score_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # Getting the values of "ID" from the content
        score_model = (content['score'])
        threshold = content['thresh']
        return score_model, threshold

    # Get list of shap_values:
    @st.cache
    def values_shap(selected_id):
        # URL of the sk_id API
        shap_values_api_url = f"{API_URL}/shap_val/?SK_ID_CURR={str(selected_id)}"
        # Requesting the API and saving the response
        response = requests.get(shap_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        shapvals = pd.DataFrame(content['shap_val_cust'].values())
        expec_vals = pd.DataFrame(content['expected_vals'].values())
        return shapvals, expec_vals

    # Get list of feature names
    @st.cache
    def feat():
        # URL of the sk_id API
        feat_api_url = f"{API_URL}/feat/"
        # Requesting the API and saving the response
        response = requests.get(feat_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        features_name = pd.Series(content['data']).values
        return features_name

    # Get the list of feature importances (according to lgbm classification model):
    @st.cache
    def get_features_importances():
        # URL of the aggregations API
        feat_imp_api_url = f"{API_URL}/feat_imp/"
        # Requesting the API and save the response
        response = requests.get(feat_imp_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        feat_imp = pd.Series(content['data']).sort_values(ascending=False)
        return feat_imp

    # Get data from 20 nearest neighbors in train set:
    @st.cache
    def get_data_neigh(selected_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        neight_data_api_url = f"{API_URL}/neigh_cust/?SK_ID_CURR={str(selected_id)}"
        # save the response of API request
        response = requests.get(neight_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        data_neig = pd.DataFrame(content['data_neigh'])
        target_neig = (pd.Series(content['y_neigh']).rename('TARGET'))
        return data_neig, target_neig

    # Get data from 1000 nearest neighbors in train set (cached):
    @st.cache
    def get_data_fivehundred_neigh(selected_id):
        fivehundred_neight_data_api_url = f"{API_URL}/fivehundred_neigh/?SK_ID_CURR={str(selected_id)}"
        # save the response of API request
        response = requests.get(fivehundred_neight_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        data_fivehundred_neig = pd.DataFrame(content['X_fivehundred_neigh'])
        x_custo = pd.DataFrame(content['X_custom'])
        target_fivehundred_neig = (pd.Series(content['y_fivehundred_neigh']).rename('TARGET'))
        return data_fivehundred_neig, target_fivehundred_neig, x_custo

    #############################################################################
    #                          Selected id
    #############################################################################
    # list of customer's ID's:
    cust_id = get_id_list()
    # Selected customer's ID:
    selected_id = st.sidebar.selectbox('Selectionner utilisateur de la liste:', cust_id, key=18)
    st.write('ID séléctionné = ', selected_id)

    ############################################################################
    #                           Graphics Functions
    ############################################################################
    # Local SHAP Graphs
    @st.cache
    def waterfall_plot(nb, ft, expected_val, shap_val):
        return shap.plots._waterfall.waterfall_legacy(expected_val, shap_val[0, :],
                                                      max_display=nb, feature_names=ft)

    # Local SHAP Graphs
    @st.cache(allow_output_mutation=True)
    def force_plot():
        shap.initjs()
        return shap.force_plot(expected_vals[0][0], shap_vals[0, :], matplotlib=True)

    # Gauge Chart
    @st.cache
    def gauge_plot(scor, th):
        scor = int(scor * 100)
        th = int(th * 100)

        if scor >= th:
            couleur_delta = 'red'
        elif scor < th:
            couleur_delta = 'Orange'

        if scor >= th:
            valeur_delta = "red"
        elif scor < th:
            valeur_delta = "green"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=scor,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Selected Customer Score", 'font': {'size': 25}},
            delta={'reference': int(th), 'increasing': {'color': valeur_delta}},
            gauge={
                'axis': {'range': [None, int(100)], 'tickwidth': 1.5, 'tickcolor': "black"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, int(th)], 'color': 'lightgreen'},
                    {'range': [int(th), int(scor)], 'color': couleur_delta}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 1,
                    'value': int(th)}}))

        fig.update_layout(paper_bgcolor="lavender", font={'color': "darkblue", 'family': "Arial"})
        return fig



    ##############################################################################
    #                         Customer's data checkbox
    ##############################################################################
    if st.sidebar.checkbox("Données Client"):
        st.markdown('Données du client séléctionné :')
        data_selected_cust, y_cust = get_selected_cust_data(selected_id)
        # data_selected_cust.columns = data_selected_cust.columns.str.split('.').str[0]
        st.write(data_selected_cust)
    ##############################################################################
    #                         Model's decision checkbox
    ##############################################################################
    if st.sidebar.checkbox("Décision du modèle", key=38):
        # Get score & threshold model
        score, threshold_model = get_score_model(selected_id)
        # Display score (default probability)
        st.write('Probabilité de défaut : {:.0f}%'.format(score * 100))
        # Display default threshold
        st.write('Seuil de défaut du modèle : {:.0f}%'.format(threshold_model * 100))  #
        # Compute decision according to the best threshold (False= loan accepted, True=loan refused)
        if score >= threshold_model:
            decision = "Crédit rejeté"
        else:
            decision = "Crédit accordé"
        st.write("Décision :", decision)
        ##########################################################################
        #              Display customer's gauge meter chart (checkbox)
        ##########################################################################
        figure = gauge_plot(score, threshold_model)
        st.write(figure)
        # Add markdown
        st.markdown('_Tracé de compteur de jauge pour le client demandeur._')
        expander = st.expander("Concernant le modèle de classification...")
        expander.write("La prédiction a été faite à l'aide du modèle de classificateur Light Gradient Boosting")
        expander.write("Le modèle par défaut est calculé pour maximiser l'air sous la courbe ROC => maximiser \
                                        Détection du taux de vrais positifs (TP) et réduction du taux de faux négatifs (FP)")
        ##########################################################################
        #                 Display local SHAP waterfall checkbox
        ##########################################################################
        if st.checkbox("Afficher l'interprétation locale de la cascade", key=25):
            with st.spinner("Graphiques en cascade SHAP en cours d'affichage..... Veuillez patienter......."):
                # Get Shap values for customer & expected values
                shap_vals, expected_vals = values_shap(selected_id)
                # index_cust = customer_ind(selected_id)
                # Get features names
                features = feat()
                # st.write(features)
                nb_features = st.slider("Nombre de variables à afficher",
                                        min_value=2,
                                        max_value=50,
                                        value=10,
                                        step=None,
                                        format=None,
                                        key=14)
                # draw the waterfall graph (only for the customer with scaling
                waterfall_plot(nb_features, features, expected_vals[0][0], shap_vals.values)

                plt.gcf()
                st.pyplot(plt.gcf())
                # Add markdown
                st.markdown('_Plot en cascade SHAP pour le client demandeur._')
                # Add details title
                expander = st.expander("Concernant le tracé en cascade du SHAP...")
                # Add explanations
                expander.write("Le graphique en cascade ci-dessus affiche \
                 explications pour la prédiction individuelle du client demandeur.\
                 Le bas d'un diagramme en cascade commence par la valeur attendue de la sortie du modèle \
                 (c'est-à-dire la valeur obtenue si aucune information (caractéristiques) n'a été fournie), puis \
                 chaque ligne montre comment la contribution positive (rouge) ou négative (bleue) de \
                 chaque caractéristique déplace la valeur de la sortie de modèle attendue sur le \
                 ensemble de données d'arrière-plan à la sortie du modèle pour cette prédiction.")

        ##########################################################################
        #              Display feature's distribution (Boxplots)
        ##########################################################################
        if st.checkbox('afficher la distribution des fonctionnalités par classe', key=20):
            st.header('Boxplots des principales caractéristiques')
            fig, ax = plt.subplots(figsize=(20, 10))
            with st.spinner('Création de boxplot en cours...veuillez patienter.....'):
                # Get Shap values for customer
                shap_vals, expected_vals = values_shap(selected_id)
                # Get features names
                features = feat()
                # Get selected columns
                disp_box_cols = get_list_display_features(features, 2, key=str(uuid4()))
                # -----------------------------------------------------------------------------------------------
                # Get tagets and data for : all customers + Applicant customer + 20 neighbors of selected customer
                # -----------------------------------------------------------------------------------------------
                # neighbors + Applicant customer :
                data_neigh, target_neigh = get_data_neigh(selected_id)
                data_fivehundred_neigh, target_fivehundred_neigh, x_customer = get_data_fivehundred_neigh(selected_id)

                x_cust, y_cust = get_selected_cust_data(selected_id)
                x_customer.columns = x_customer.columns.str.split('.').str[0]
                # Target impuatation (0 : 'repaid (....), 1 : not repaid (....)
                # -------------------------------------------------------------
                target_neigh = target_neigh.replace({0: 'repaid (neighbors)',
                                                     1: 'not repaid (neighbors)'})
                target_fivehundred_neigh = target_fivehundred_neigh.replace({0: 'repaid (neighbors)',
                                                                       1: 'not repaid (neighbors)'})
                y_cust = y_cust.replace({0: 'repaid (customer)',
                                         1: 'not repaid (customer)'})

                # ------------------------------
                # Get 500 neighbors personal data
                # ------------------------------
                df_fivehundred_neigh = pd.concat([data_fivehundred_neigh[disp_box_cols], target_fivehundred_neigh], axis=1)
                df_melt_fivehundred_neigh = df_fivehundred_neigh.reset_index()
                df_melt_fivehundred_neigh.columns = ['index'] + list(df_melt_fivehundred_neigh.columns)[1:]
                df_melt_fivehundred_neigh = df_melt_fivehundred_neigh.melt(id_vars=['index', 'TARGET'],
                                                                     value_vars=disp_box_cols,
                                                                     var_name="variables",  # "variables",
                                                                     value_name="values")

                sns.boxplot(data=df_melt_fivehundred_neigh, x='variables', y='values',
                            hue='TARGET', linewidth=1, width=0.4,
                            palette=['tab:green', 'tab:red'], showfliers=False,
                            saturation=0.5, ax=ax)

                # ------------------------------
                # Get 20 neighbors personal data
                # ------------------------------
                df_neigh = pd.concat([data_neigh[disp_box_cols], target_neigh], axis=1)
                df_melt_neigh = df_neigh.reset_index()
                df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
                df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],
                                                   value_vars=disp_box_cols,
                                                   var_name="variables",  # "variables",
                                                   value_name="values")

                sns.swarmplot(data=df_melt_neigh, x='variables', y='values', hue='TARGET', linewidth=1,
                              palette=['darkgreen', 'darkred'], marker='o', size=15, edgecolor='k', ax=ax)

                # -----------------------
                # Applicant customer data
                # -----------------------
                df_selected_cust = pd.concat([x_customer[disp_box_cols], y_cust], axis=1)
                # st.write("df_sel_cust :", df_sel_cust)
                df_melt_sel_cust = df_selected_cust.reset_index()
                df_melt_sel_cust.columns = ['index'] + list(df_melt_sel_cust.columns)[1:]
                df_melt_sel_cust = df_melt_sel_cust.melt(id_vars=['index', 'TARGET'],
                                                         value_vars=disp_box_cols,
                                                         var_name="variables",
                                                         value_name="values")

                sns.swarmplot(data=df_melt_sel_cust, x='variables', y='values',
                              linewidth=1, color='y', marker='o', size=20,
                              edgecolor='k', label='applicant customer', ax=ax)

                # legend
                h, _ = ax.get_legend_handles_labels()
                ax.legend(handles=h[:5])

                plt.xticks(rotation=20, ha='right')
                plt.show()

                st.write(fig)  # st.pyplot(fig) # the same

                plt.xticks(rotation=20, ha='right')
                plt.show()

                st.markdown('_Dispersion of the main features for random sample,\
                20 nearest neighbors and applicant customer_')

                expander = st.expander("Concerning the dispersion graph...")
                expander.write("These boxplots show the dispersion of the preprocessed features values\
                used by the model to make a prediction. The green boxplot are for the customers that repaid \
                their loan, and red boxplots are for the customers that didn't repay it.Over the boxplots are\
                superimposed (markers) the values\
                of the features for the 20 nearest neighbors of the applicant customer in the training set. The \
                color of the markers indicate whether or not these neighbors repaid their loan. \
                Values for the applicant customer are superimposed in yellow.")


if __name__ == "__main__":
    main()
