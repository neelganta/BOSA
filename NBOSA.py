import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


import sklearn

# advanced algorthms
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

#import decisiontreeclassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
#import logisticregression classifier
from sklearn.linear_model import LogisticRegression
#import knn classifier
from sklearn.neighbors import KNeighborsClassifier

#for validating your classification model
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# feature selection
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


#  Basketball Operations Seasonal Assistant
st.title('Brooklyn Nets BOSA Project')
st.markdown('_Please see left sidebar for more details._')

df = pd.read_csv('https://raw.githubusercontent.com/neelganta/neel_project/master/BOSA_Train.csv') #, delimiter= None, error_bad_lines=False, header = 0)
# convert categorical variables to dummy variables
df =  pd.get_dummies(df, columns=["position"], prefix=["position"], drop_first=True)
#mapping or replacing
df = df.replace({'NBA_Success': 'No'}, {'NBA_Success': '0'})
df = df.replace({'NBA_Success': 'Yes'}, {'NBA_Success': '1'})
df['NBA_Success'] = pd.to_numeric(df['NBA_Success'])

score = pd.read_csv('https://raw.githubusercontent.com/neelganta/neel_project/master/BOSA_Test.csv')
score =  pd.get_dummies(score, columns=["position"], prefix=["position"], drop_first=True)
score['position_F'] = 0
score['position_G'] = 0
score['position_PG/SG/SF'] = 0


if st.checkbox('Show Draft Prospect Dataframe'):
    st.write(score)
 

# declare X variables and y variable

y = df['NBA_Success']
X = df.drop(['NBA_Success', 'id'], axis=1)

st.subheader('Machine Learning Models')
alg = ['Select Algorithm', 'Chi-Squared Logistic Regression', 'Random Forest Decision Tree']
classifier = st.selectbox('Which classification algorithm would you like to use?', alg)

if classifier!= 'Select Algorithm' and classifier =='Chi-Squared Logistic Regression':
    X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
    X_new_df = pd.DataFrame(X_new)
    X_new_df = X_new_df.rename(columns={0: 'Ranking', 1: 'ASTP', 2: 'BLKP'})
    X_train, X_test, y_train, y_test = train_test_split(X_new,y, test_size = 0.3, random_state = 0)
    lr = LogisticRegression(C = 2.195254015709299, penalty = "l1", solver='liblinear')
    lr.fit(X_train, y_train)
    acc = lr.score(X_test, y_test)
    st.markdown('**Accuracy Score**')
    st.write(acc)
    auc = (metrics.roc_auc_score(y_test, lr.predict(X_test)))
    st.markdown('**AUC - ROC Score **')
    st.write(auc)
    pred_dtc = lr.predict(X_test)
    cm =confusion_matrix(y_test,pred_dtc)
    cm = pd.DataFrame(cm)
    cm.columns = ['Neg', 'Pos']
    Row_list =[] 
    
    # Iterate over each row 
    for index, rows in cm.iterrows(): 
        # Create list for the current row 
        my_list =[rows.Neg, rows.Pos] 
        
        # append the list to the final list 
        Row_list.append(my_list) 
        
    z = Row_list

    x =['No NBA Success', 'NBA Success']
    y=['No NBA Success', 'NBA Success']
    # change each element of z to type string for annotations
    z_text = [[str(x) for x in y] for y in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    # add title
    fig.update_layout(title_text='<i><b>Confusion Matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Actual Success",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Predicted Success",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))
    # add colorbar
    fig['data'][0]['showscale'] = True
    st.plotly_chart(fig)
    predict_new = score[['Ranking', 'ASTP', 'BLKP']].copy()
    probs = lr.predict_proba(predict_new)
    probs = pd.DataFrame(probs, columns=['Probability of No NBA Success', 'Probability of NBA Success'])
    probs['id'] = score['id']
    probs = probs[['id', 'Probability of No NBA Success', 'Probability of NBA Success']]
    ids = score['id'].tolist()
    ids.insert(0, 'Draft Prospects by ID') 
    st.subheader('Predicting Prospect Success')
    selector = st.selectbox("Select a Draft Prospect's ID to predict their NBA success.", ids)     
    if selector != 'Draft Prospects by ID':
        a = score.loc[score['id'] == selector]
        a = a[['Ranking', 'ASTP', 'BLKP']] #most accurate
        a = lr.predict_proba(a)
        b = pd.DataFrame(a).round(3)
        b.columns=['Probability of No NBA Success', 'Probability of NBA Success']
        c=b.T.reset_index()
        c.columns=['NBA Success','Probability']
        fig = px.bar(c, x='NBA Success', y='Probability', text= 'Probability')
        fig.update_layout(title='Draft Prospect '+ str(selector))
        st.plotly_chart(fig)
        good = b.values[0][1]
        if good > .5:
            st.success("Draft Prospect " + str(selector) +" is "+ str((good*100).round(2)) + '%' +" likely to be successful.")
        elif good < .5:
            st.error("Draft Prospect " + str(selector) +" is "+ str(((1-good)*100).round(2))+ '%' +" likely to be unsuccessful.")

        if st.checkbox('Show all Draft Prospect Probabilities'):
            probs
    
        st.subheader('Create and Predict your own Draft Prospect')
        rank = st.slider("Choose the Ranking of your player: ", min_value=1, max_value=60, value=30, step=1)
        ast = st.slider("Choose the Assist Percentage of your player: ", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
        blk = st.slider("Choose the Block Percentage of your player: ", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        user_prediction_data = [[rank, ast, blk]]
        if st.button('PREDICT'):
            a = lr.predict_proba(user_prediction_data)
            b = pd.DataFrame(a).round(3)
            b.columns=['Probability of No NBA Success', 'Probability of NBA Success']
            c=b.T.reset_index()
            c.columns=['NBA Success','Probability']
            fig = px.bar(c, x='NBA Success', y='Probability', text= 'Probability')
            fig.update_layout(title='Your Draft Prospect')
            st.plotly_chart(fig)
            good = b.values[0][1]
            if good > .5:
                st.success("Your Draft Prospect " + " is "+ str((good*100).round(2))+ '%' +" likely to be successful.")
                st.balloons()
            elif good < .5:
                st.error("Your Draft Prospect " + " is "+ str(((1-good)*100).round(2))+ '%' +" likely to be unsuccessful.")


if classifier!= 'Select Algorithm' and classifier =='Random Forest Decision Tree':
    X_clf_new_df = df[['Ranking', 'ASTP', 'Age', 'BLKP']].copy() #most accurate
    X_clf_new_df = X_clf_new_df.rename(columns={0: 'Ranking', 1: 'ASTP', 2: 'Age', 3: 'BLKP'})
    X_train, X_test, y_train, y_test = train_test_split(X_clf_new_df,y, test_size = 0.3, random_state = 0)
    lr = LogisticRegression(C = 2.195254015709299, penalty = "l1", solver='liblinear')
    lr.fit(X_train, y_train)
    acc = lr.score(X_test, y_test)
    st.markdown('**Accuracy Score**')
    st.write(acc)
    auc = (metrics.roc_auc_score(y_test, lr.predict(X_test)))
    st.markdown('**AUC - ROC Score **')
    st.write(auc)
    pred_clf = lr.predict(X_test)
    cm=confusion_matrix(y_test,pred_clf)
    cm = pd.DataFrame(cm)
    cm.columns = ['Neg', 'Pos']
    Row_list =[] 
    
    # Iterate over each row 
    for index, rows in cm.iterrows(): 
        # Create list for the current row 
        my_list =[rows.Neg, rows.Pos] 
        
        # append the list to the final list 
        Row_list.append(my_list) 
        
    z = Row_list

    x =['No NBA Success', 'NBA Success']
    y=['No NBA Success', 'NBA Success']
    # change each element of z to type string for annotations
    z_text = [[str(x) for x in y] for y in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    # add title
    fig.update_layout(title_text='<i><b>Confusion Matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Actual Success",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Predicted Success",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))
    # add colorbar
    fig['data'][0]['showscale'] = True
    st.plotly_chart(fig)
    predict_new_clf = score[['Ranking', 'ASTP', 'Age', 'BLKP']].copy()
    probs = lr.predict_proba(predict_new_clf)
    probs = pd.DataFrame(probs, columns=['Probability of No NBA Success', 'Probability of NBA Success'])
    probs['id'] = score['id']
    probs = probs[['id', 'Probability of No NBA Success', 'Probability of NBA Success']]
    ids = score['id'].tolist()
    ids.insert(0, 'Draft Prospects by ID') 
    st.subheader('Predicting Prospect Success')
    selector = st.selectbox("Select a Draft Prospect's ID to predict their NBA success.", ids)     
    if selector != 'Draft Prospects by ID':
        a = score.loc[score['id'] == selector]
        a = a[['Ranking', 'ASTP', 'Age', 'BLKP']] #most accurate
        a = lr.predict_proba(a)
        b = pd.DataFrame(a).round(3)
        b.columns=['Probability of No NBA Success', 'Probability of NBA Success']
        c=b.T.reset_index()
        c.columns=['NBA Success','Probability']
        fig = px.bar(c, x='NBA Success', y='Probability', text= 'Probability')
        fig.update_layout(title='Draft Prospect '+ str(selector))
        st.plotly_chart(fig)
        good = b.values[0][1]
        if good > .5:
            st.success("Draft Prospect " + str(selector) +" is "+ str((good*100).round(2)) + '%' +" likely to be successful.")
        elif good < .5:
            st.error("Draft Prospect " + str(selector) +" is "+ str(((1-good)*100).round(2))+ '%' +" likely to be unsuccessful.")
        if st.checkbox('Show all Draft Prospect Probabilities'):
            probs
        st.subheader('Create and Predict your own Draft Prospect')
        rank = st.slider("Choose the Ranking of your player: ", min_value=1, max_value=60, value=30, step=1)
        age = st.slider("Choose the Age of your player: ", min_value=18.0, max_value=30.0, value=20.0, step=0.1)
        ast = st.slider("Choose the Assist Percentage of your player: ", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
        blk = st.slider("Choose the Block Percentage of your player: ", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        user_prediction_data = [[rank, ast, age, blk]]
        if st.button('PREDICT'):
            a = lr.predict_proba(user_prediction_data)
            b = pd.DataFrame(a).round(3)
            b.columns=['Probability of No NBA Success', 'Probability of NBA Success']
            c=b.T.reset_index()
            c.columns=['NBA Success','Probability']
            fig = px.bar(c, x='NBA Success', y='Probability', text= 'Probability')
            fig.update_layout(title='Your Draft Prospect ')
            st.plotly_chart(fig)
            good = b.values[0][1]
            if good > .5:
                st.success("Your Draft Prospect " + " is "+ str((good*100).round(2))+ '%' +" likely to be successful.")
                st.balloons()
            elif good < .5:
                st.error("Your Draft Prospect " + " is "+ str(((1-good)*100).round(2))+ '%' +" likely to be unsuccessful.")




st.markdown('_Presented by Neel Ganta._')

# st.sidebar.markdown('**ABOUT THE NBA LINEUP MACHINE:**  The _NBA Lineup Machine_ was first incepted roughly one year ago while Neel Ganta was pondering the current lineup problem in the NBA. Should teams go small? Three shooters? Five? How can we see what our team would look like with a player _before_ trading for him? Seeing a problem and no publicly available solution, Neel decided to create what could be the next big GM tool. Please enjoy the _NBA Lineup Machine_ which allows you to input **any** five players in the NBA, and predicts an overall Net Rating for the lineup.')

st.sidebar.markdown('**ABOUT THE BOSA PROJECT:**  After creating the _[NBA Lineup Machine](https://nba-lineup-machine.herokuapp.com)_, which allows the user to predict the Net Rating of any lineup possible in the current NBA, I developed experience in the web application world. With that experience, I decided to create an interactive web application for you to interact with. Please enjoy all the features in this application, spanning from displaying different dataframes, selecting your machine learning algorithm, interacting with visualizations, selecting which draft prospect to predict success for, and creating your own player to predict success on.')
st.sidebar.markdown('_**[Chi Squared Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html)**: The Chi Square test measures dependence between different variables, so using this function “weeds out” the variables that are the most likely to be independent of NBA Success and therefore irrelevant for classification. Once the three most important dependent variables towards NBA Success were found (Ranking, ASTP, BLKP), I used just these three variables in a logistic regression. A logistic regression takes any number of x variables as input to try and predict a y variable, which in this case is NBA Success._')
st.sidebar.markdown('_**[Random Forest Decision Tree](https://towardsdatascience.com/decision-trees-and-random-forests-df0c3123f991)**: A decision tree can be used as a classification tool that answers sequential questions, moving further down the tree given the previous answers. One decision tree alone may be prone to overfitting, and that is where the random forest comes in to play. A random forest is a large collection of random decision trees whose results are aggregated into one final result. They have an ability to limit overfitting without substantially increasing error due to bias, which is why they are such powerful models._')
st.sidebar.markdown('_**[Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html?highlight=metrics.accuracy_score#sklearn.metrics.accuracy_score)**: used to compare set of predicted labels for a sample to the corresponding set of labels in y_true, or in other words, measures the similarity between the actual and predicted datasets._ ')
st.sidebar.markdown("_**[AUC - ROC Score](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)**: one of the most important metrics for evaluating a classification model, the AUC ROC (Area Under the Curve of the Receiver Operating Characteristics) tells how much a model is capable of distinguishing between classes. The higher the AUC score, the better the model is at predicting classes correctly. In our terms, the higher the AUC, the better the model is at distinguishing between prospects with no NBA Success and NBA Success._ ")
st.sidebar.markdown('_**[Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)**: Actual success values are on the x-axis, and predicted success values are on the y-axis. For example, the bottom left quadrant shows how many predicted unsuccessful NBA players were actually unsuccessful. We call this number a "True Negative". To the right of this quadrant, we see how many players were predicted as unsuccessful, but actually were successful. We call this number a "False Negative". This exact same methodology applies to the upper two quadrants, and this is extermely useful for interpreting and measuring accuracy as well as the AUC-ROC score. _')
st.sidebar.markdown('_Presented by Neel Ganta._')
