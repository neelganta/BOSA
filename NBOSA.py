import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


import sklearn

# advanced algorthms
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

#import decisiontreeclassifier
from sklearn.tree import DecisionTreeClassifier
#import logisticregression classifier
from sklearn.linear_model import LogisticRegression
#import knn classifier
from sklearn.neighbors import KNeighborsClassifier

#for validating your classification model
from sklearn.model_selection import  cross_val_score, train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# feature selection
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
alg = ['Select Algorithm', 'Decision Tree Classifier', 'Chi-Squared - Logistic Regression', 'Gradient Boosted Decision Trees', 
'Random Forest Ensemble Decision Trees', 'Random Forest Ensemble Decision Trees - Logistic Regression', 'Support Vector Machine', 'Neural Network']
classifier = st.selectbox('Which classification algorithm would you like to use?', alg)

if classifier!= 'Select Algorithm' and classifier =='Chi-Squared - Logistic Regression':
    X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_new,y, test_size = 0.3, random_state = 0)
    clf = LogisticRegression(C = 2.195254015709299, penalty = "l1", solver='liblinear')
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    st.markdown('**Accuracy Score**')
    st.write(acc)
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
    st.markdown('**Cross Validated Accuracy Score**')
    st.write(scores.mean())
    auc = (metrics.roc_auc_score(y_test, clf.predict(X_test)))
    st.markdown('**AUC - ROC Score **')
    st.write(auc)
    
    st.image('https://raw.githubusercontent.com/neelganta/neel_project/master/CSLRCM.png')
    predict_new = score[['Ranking', 'ASTP', 'BLKP']].copy()
    probs = clf.predict_proba(predict_new)
    probs = pd.DataFrame(probs, columns=['Probability of No NBA Success', 'Probability of NBA Success'])
    probs['id'] = score['id']
    probs = probs[['id', 'Probability of No NBA Success', 'Probability of NBA Success']]
    ids = score['id'].tolist()
    ids.insert(0, 'Draft Prospects by ID') 
    st.subheader('Predicting Prospect Success with Chi-Squared - Logistic Regression')
    selector = st.selectbox("Select a Draft Prospect's ID to predict their NBA success.", ids)     
    if selector != 'Draft Prospects by ID':
        a = score.loc[score['id'] == selector]
        a = a[['Ranking', 'ASTP', 'BLKP']] #most accurate
        a = clf.predict_proba(a)
        b = pd.DataFrame(a).round(4)
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
        
        st.subheader('Create your own Scatterplots of Draft Prospect Correlations')
        scatter = probs.set_index('id').join(score.set_index('id'))
        scatter = scatter.reset_index()
        xaxis = st.selectbox('Select which feature you want on your x-axis.', scatter.columns)
        yaxis = st.selectbox('Select which feature you want on your y-axis.', scatter.columns)

        if(xaxis != 'id' and yaxis !='id'):
            fig = px.scatter(scatter, x = xaxis, y =yaxis, hover_name = 'id', hover_data = [xaxis], trendline= 'ols' , color = yaxis)
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects ' + xaxis +' vs. ' + yaxis,
                yaxis_title= yaxis
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)
        else:
            fig = px.scatter(scatter, x = "Ranking", y ='Probability of No NBA Success', hover_name = 'id', hover_data = ['Probability of No NBA Success'],  trendline = 'ols', color = 'Probability of No NBA Success')
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects Ranking vs. Probability of No NBA Success',
                yaxis_title='Probability of No NBA Success'
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)
        if st.checkbox('Show all Draft Prospect Probabilities'):
            probs
    
        st.subheader("Create and Predict your own Draft Prospect with Chi-Squared - Logistic Regression")
        rank = st.slider("Choose the Ranking of your player: ", min_value=1, max_value=60, value=30, step=1)
        ast = st.slider("Choose the Assist Percentage of your player: ", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
        blk = st.slider("Choose the Block Percentage of your player: ", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        user_prediction_data = [[rank, ast, blk]]
        if st.button('PREDICT'):
            a = clf.predict_proba(user_prediction_data)
            b = pd.DataFrame(a).round(4)
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


elif classifier!= 'Select Algorithm' and classifier =='Random Forest Ensemble Decision Trees - Logistic Regression':
    X_clf_new_df = df[['Ranking', 'ASTP', 'Age', 'BLKP']].copy() #most accurate
    X_clf_new_df = X_clf_new_df.rename(columns={0: 'Ranking', 1: 'ASTP', 2: 'Age', 3: 'BLKP'})
    X_train, X_test, y_train, y_test = train_test_split(X_clf_new_df,y, test_size = 0.3, random_state = 0)
    clf = LogisticRegression(C = 2.195254015709299, penalty = "l1", solver='liblinear')
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    st.markdown('**Accuracy Score**')
    st.write(acc)
    auc = (metrics.roc_auc_score(y_test, clf.predict(X_test)))
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
    st.markdown('**Cross Validated Accuracy Score**')
    st.write(scores.mean())
    st.markdown('**AUC - ROC Score **')
    st.write(auc)

    st.image('https://raw.githubusercontent.com/neelganta/neel_project/master/RFELRCM.png')
            
    predict_new_dt = score[['Ranking', 'ASTP', 'Age', 'BLKP']].copy()
    probs = clf.predict_proba(predict_new_dt)
    probs = pd.DataFrame(probs, columns=['Probability of No NBA Success', 'Probability of NBA Success'])
    probs['id'] = score['id']
    probs = probs[['id', 'Probability of No NBA Success', 'Probability of NBA Success']]
    ids = score['id'].tolist()
    ids.insert(0, 'Draft Prospects by ID') 
    st.subheader('Predicting Prospect Success with Random Forest Ensemble Decision Trees - Logistic Regression')
    selector = st.selectbox("Select a Draft Prospect's ID to predict their NBA success.", ids)     
    if selector != 'Draft Prospects by ID':
        a = score.loc[score['id'] == selector]
        a = a[['Ranking', 'ASTP', 'Age', 'BLKP']] #most accurate
        a = clf.predict_proba(a)
        b = pd.DataFrame(a).round(4)
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

        st.subheader('Create your own Scatterplots of Draft Prospect Correlations')
        scatter = probs.set_index('id').join(score.set_index('id'))
        scatter = scatter.reset_index()
        xaxis = st.selectbox('Select which feature you want on your x-axis.', scatter.columns)
        yaxis = st.selectbox('Select which feature you want on your y-axis.', scatter.columns)

        if(xaxis != 'id' and yaxis !='id'):
            fig = px.scatter(scatter, x = xaxis, y =yaxis, hover_name = 'id', hover_data = [xaxis], trendline= 'ols' , color = yaxis)
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects ' + xaxis +' vs. ' + yaxis,
                yaxis_title= yaxis
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)
        else:
            fig = px.scatter(scatter, x = "Ranking", y ='Probability of No NBA Success', hover_name = 'id', hover_data = ['Probability of No NBA Success'],  trendline = 'ols', color = 'Probability of No NBA Success')
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects Ranking vs. Probability of No NBA Success',
                yaxis_title='Probability of No NBA Success'
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)

        if st.checkbox('Show all Draft Prospect Probabilities'):
            probs
        st.subheader("Create and Predict your own Draft Prospect with Random Forest Ensemble Decision Trees - Logistic Regression")
        rank = st.slider("Choose the Ranking of your player: ", min_value=1, max_value=60, value=30, step=1)
        age = st.slider("Choose the Age of your player: ", min_value=18.0, max_value=30.0, value=20.0, step=0.1)
        ast = st.slider("Choose the Assist Percentage of your player: ", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
        blk = st.slider("Choose the Block Percentage of your player: ", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        user_prediction_data = [[rank, ast, age, blk]]
        if st.button('PREDICT'):
            a = clf.predict_proba(user_prediction_data)
            b = pd.DataFrame(a).round(4)
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


elif classifier!= 'Select Algorithm' and classifier =='Random Forest Ensemble Decision Trees':
    X = df[['Ranking', 'Age', 'ASTP', 'STLP', 'PER']].copy()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
    clf = RandomForestClassifier(n_estimators=1000)    #building 1000 decision trees
    clf=clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    st.markdown('**Accuracy Score**')
    st.write(acc)
    auc = (metrics.roc_auc_score(y_test, clf.predict(X_test)))
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
    st.markdown('**Cross Validated Accuracy Score**')
    st.write(scores.mean())
    st.markdown('**AUC - ROC Score **')
    st.write(auc)

    st.image('https://raw.githubusercontent.com/neelganta/neel_project/master/RFECM.png')
    
    predict_new_dt = score[['Ranking', 'Age', 'ASTP', 'STLP', 'PER']].copy()
    probs = clf.predict_proba(predict_new_dt)
    probs = pd.DataFrame(probs, columns=['Probability of No NBA Success', 'Probability of NBA Success'])
    probs['id'] = score['id']
    probs = probs[['id', 'Probability of No NBA Success', 'Probability of NBA Success']]
    ids = score['id'].tolist()
    ids.insert(0, 'Draft Prospects by ID') 
    st.subheader('Predicting Prospect Success with Random Forest Ensemble Decision Trees')
    selector = st.selectbox("Select a Draft Prospect's ID to predict their NBA success.", ids)     
    if selector != 'Draft Prospects by ID':
        a = score.loc[score['id'] == selector]
        a = a[['Ranking', 'Age', 'ASTP', 'STLP', 'PER']] 
        a = clf.predict_proba(a)
        b = pd.DataFrame(a).round(4)
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

        st.subheader('Create your own Scatterplots of Draft Prospect Correlations')
        scatter = probs.set_index('id').join(score.set_index('id'))
        scatter = scatter.reset_index()
        xaxis = st.selectbox('Select which feature you want on your x-axis.', scatter.columns)
        yaxis = st.selectbox('Select which feature you want on your y-axis.', scatter.columns)

        if(xaxis != 'id' and yaxis !='id'):
            fig = px.scatter(scatter, x = xaxis, y =yaxis, hover_name = 'id', hover_data = [xaxis], trendline= 'ols' , color = yaxis)
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects ' + xaxis +' vs. ' + yaxis,
                yaxis_title= yaxis
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)
        else:
            fig = px.scatter(scatter, x = "Ranking", y ='Probability of No NBA Success', hover_name = 'id', hover_data = ['Probability of No NBA Success'],  trendline = 'ols', color = 'Probability of No NBA Success')
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects Ranking vs. Probability of No NBA Success',
                yaxis_title='Probability of No NBA Success'
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)

        if st.checkbox('Show all Draft Prospect Probabilities'):
            probs
        st.subheader("Create and Predict your own Draft Prospect with Random Forest Ensemble Decision Trees")
        rank = st.slider("Choose the Ranking of your player: ", min_value=1, max_value=60, value=30, step=1)
        age = st.slider("Choose the Age of your player: ", min_value=18.0, max_value=30.0, value=20.0, step=0.1)
        ast = st.slider("Choose the Assist Percentage of your player: ", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
        stl = st.slider("Choose the Steal Percentage of your player: ", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        per = st.slider("Choose the Player Efficiency Rating of your player: ", min_value=0.0, max_value=50.0, value=15.0, step=0.1)

        user_prediction_data = [[rank, age, ast, stl, per]]
        if st.button('PREDICT'):
            a = clf.predict_proba(user_prediction_data)
            b = pd.DataFrame(a).round(4)
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


### DECISION TREE CLASSIFIER

elif classifier!= 'Select Algorithm' and classifier =='Decision Tree Classifier':
    X_new = df[['Ranking', 'FTp', 'TOVP', 'ASTP']].copy()
    X_train, X_test, y_train, y_test = train_test_split(X_new,y, test_size = 0.3, random_state = 42)
    # You can make a simpler decision tree ... name the model "dt_simple" (max_depth=3, min_samples_leaf=5)
    dt_simple = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
    # Train a decision tree model
    dt_simple = dt_simple.fit(X_train, y_train)
    acc = dt_simple.score(X_test, y_test)
    st.markdown('**Accuracy Score**')
    st.write(acc)
    auc = (metrics.roc_auc_score(y_test, dt_simple.predict(X_test)))
    scores = cross_val_score(dt_simple, X, y, scoring='accuracy', cv=10)
    st.markdown('**Cross Validated Accuracy Score**')
    st.write(scores.mean())
    st.markdown('**AUC - ROC Score **')
    st.write(auc)

    st.image('https://raw.githubusercontent.com/neelganta/neel_project/master/DTCM.png')
    
    predict_new_dt = score[['Ranking', 'FTp', 'TOVP', 'ASTP']].copy()
    probs = dt_simple.predict_proba(predict_new_dt)
    probs = pd.DataFrame(probs, columns=['Probability of No NBA Success', 'Probability of NBA Success'])
    probs['id'] = score['id']
    probs = probs[['id', 'Probability of No NBA Success', 'Probability of NBA Success']]
    ids = score['id'].tolist()
    ids.insert(0, 'Draft Prospects by ID') 
    st.subheader('Predicting Prospect Success with Decision Tree Classifier')
    selector = st.selectbox("Select a Draft Prospect's ID to predict their NBA success.", ids)     
    if selector != 'Draft Prospects by ID':
        a = score.loc[score['id'] == selector]
        a = a[['Ranking', 'ASTP', 'Age', 'BLKP']] #most accurate
        a = dt_simple.predict_proba(a)
        b = pd.DataFrame(a).round(4)
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

        st.subheader('Create your own Scatterplots of Draft Prospect Correlations')
        scatter = probs.set_index('id').join(score.set_index('id'))
        scatter = scatter.reset_index()
        xaxis = st.selectbox('Select which feature you want on your x-axis.', scatter.columns)
        yaxis = st.selectbox('Select which feature you want on your y-axis.', scatter.columns)

        if(xaxis != 'id' and yaxis !='id'):
            fig = px.scatter(scatter, x = xaxis, y =yaxis, hover_name = 'id', hover_data = [xaxis], trendline= 'ols' , color = yaxis)
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects ' + xaxis +' vs. ' + yaxis,
                yaxis_title= yaxis
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)
        else:
            fig = px.scatter(scatter, x = "Ranking", y ='Probability of No NBA Success', hover_name = 'id', hover_data = ['Probability of No NBA Success'],  trendline = 'ols', color = 'Probability of No NBA Success')
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects Ranking vs. Probability of No NBA Success',
                yaxis_title='Probability of No NBA Success'
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)

        if st.checkbox('Show all Draft Prospect Probabilities'):
            probs

        st.subheader("Create and Predict your own Draft Prospect with  Decision Tree Classifier")
        rank = st.slider("Choose the Ranking of your player: ", min_value=1, max_value=60, value=30, step=1)
        ftp = st.slider("Choose the Free Throw percentage of your player: ", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        ast = st.slider("Choose the Assist Percentage of your player: ", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
        tov = st.slider("Choose the Turnover Percentage of your player: ", min_value=3.0, max_value=30.0, value=10.0, step=0.1)
        user_prediction_data = [[rank, ftp, tov, ast]]
        if st.button('PREDICT'):
            a = dt_simple.predict_proba(user_prediction_data)
            b = pd.DataFrame(a).round(4)
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
            st.subheader('Decision Tree')
            st.image('https://raw.githubusercontent.com/neelganta/neel_project/master/BOSA_dt.png')

elif classifier!= 'Select Algorithm' and classifier =='Gradient Boosted Decision Trees':
    X = df[['Ranking', 'STLP', 'FTp', '3PAr', 'Age']].copy()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
    clf = GradientBoostingClassifier(n_estimators=1000)    #building 1000 decision trees
    clf=clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    st.markdown('**Accuracy Score**')
    st.write(acc)
    auc = (metrics.roc_auc_score(y_test, clf.predict(X_test)))
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
    st.markdown('**Cross Validated Accuracy Score**')
    st.write(scores.mean())
    st.markdown('**AUC - ROC Score **')
    st.write(auc)

    st.image('https://raw.githubusercontent.com/neelganta/neel_project/master/GBCM.png')
    
    predict_new_dt = score[['Ranking', 'STLP', 'FTp', '3PAr', 'Age']].copy()
    probs = clf.predict_proba(predict_new_dt)
    probs = pd.DataFrame(probs, columns=['Probability of No NBA Success', 'Probability of NBA Success'])
    probs['id'] = score['id']
    probs = probs[['id', 'Probability of No NBA Success', 'Probability of NBA Success']]
    ids = score['id'].tolist()
    ids.insert(0, 'Draft Prospects by ID') 
    st.subheader('Predicting Prospect Success with Gradient Boosted Decision Trees')
    selector = st.selectbox("Select a Draft Prospect's ID to predict their NBA success.", ids)     
    if selector != 'Draft Prospects by ID':
        a = score.loc[score['id'] == selector]
        a = a[['Ranking', 'STLP', 'FTp', '3PAr', 'Age']] 
        a = clf.predict_proba(a)
        b = pd.DataFrame(a).round(4)
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

        st.subheader('Create your own Scatterplots of Draft Prospect Correlations')
        scatter = probs.set_index('id').join(score.set_index('id'))
        scatter = scatter.reset_index()
        xaxis = st.selectbox('Select which feature you want on your x-axis.', scatter.columns)
        yaxis = st.selectbox('Select which feature you want on your y-axis.', scatter.columns)

        if(xaxis != 'id' and yaxis !='id'):
            fig = px.scatter(scatter, x = xaxis, y =yaxis, hover_name = 'id', hover_data = [xaxis], trendline= 'ols' , color = yaxis)
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects ' + xaxis +' vs. ' + yaxis,
                yaxis_title= yaxis
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)
        else:
            fig = px.scatter(scatter, x = "Ranking", y ='Probability of No NBA Success', hover_name = 'id', hover_data = ['Probability of No NBA Success'],  trendline = 'ols', color = 'Probability of No NBA Success')
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects Ranking vs. Probability of No NBA Success',
                yaxis_title='Probability of No NBA Success'
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)

        if st.checkbox('Show all Draft Prospect Probabilities'):
            probs

        st.subheader("Create and Predict your own Draft Prospect with Gradient Boosted Decision Trees")
        rank = st.slider("Choose the Ranking of your player: ", min_value=1, max_value=60, value=30, step=1)
        age = st.slider("Choose the Age of your player: ", min_value=18.0, max_value=30.0, value=20.0, step=0.1)
        ft = st.slider("Choose the Free Throw Percentage of your player: ", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        stl = st.slider("Choose the Steal Percentage of your player: ", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        tpa = st.slider("Choose the Three Point Attempt Rate of your player: ", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

        user_prediction_data = [[rank, stl, ft, tpa, age]]
        if st.button('PREDICT'):
            a = clf.predict_proba(user_prediction_data)
            b = pd.DataFrame(a).round(4)
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


elif classifier!= 'Select Algorithm' and classifier =='Support Vector Machine':
    X = df[['Ranking', 'STLP', 'FTp', '3PAr', 'Age']].copy()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
    clf = SVC(gamma='scale', probability=True)    #building 1000 decision trees
    clf=clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    st.markdown('**Accuracy Score**')
    st.write(acc)
    auc = (metrics.roc_auc_score(y_test, clf.predict(X_test)))
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
    st.markdown('**Cross Validated Accuracy Score**')
    st.write(scores.mean())
    st.markdown('**AUC - ROC Score **')
    st.write(auc)

    st.image('https://raw.githubusercontent.com/neelganta/neel_project/master/SVMCM.png')

    predict_new_dt = score[['Ranking', 'STLP', 'FTp', '3PAr', 'Age']].copy()
    probs = clf.predict_proba(predict_new_dt)
    probs = pd.DataFrame(probs, columns=['Probability of No NBA Success', 'Probability of NBA Success'])
    probs['id'] = score['id']
    probs = probs[['id', 'Probability of No NBA Success', 'Probability of NBA Success']]
    ids = score['id'].tolist()
    ids.insert(0, 'Draft Prospects by ID') 
    st.subheader('Predicting Prospect Success with Support Vector Machine')
    selector = st.selectbox("Select a Draft Prospect's ID to predict their NBA success.", ids)     
    if selector != 'Draft Prospects by ID':
        a = score.loc[score['id'] == selector]
        a = a[['Ranking', 'STLP', 'FTp', '3PAr', 'Age']] 
        a = clf.predict_proba(a)
        b = pd.DataFrame(a).round(4)
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
    
        st.subheader('Create your own Scatterplots of Draft Prospect Correlations')
        scatter = probs.set_index('id').join(score.set_index('id'))
        scatter = scatter.reset_index()
        xaxis = st.selectbox('Select which feature you want on your x-axis.', scatter.columns)
        yaxis = st.selectbox('Select which feature you want on your y-axis.', scatter.columns)

        if(xaxis != 'id' and yaxis !='id'):
            fig = px.scatter(scatter, x = xaxis, y =yaxis, hover_name = 'id', hover_data = [xaxis], trendline= 'ols' , color = yaxis)
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects ' + xaxis +' vs. ' + yaxis,
                yaxis_title= yaxis
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)
        else:
            fig = px.scatter(scatter, x = "Ranking", y ='Probability of No NBA Success', hover_name = 'id', hover_data = ['Probability of No NBA Success'],  trendline = 'ols', color = 'Probability of No NBA Success')
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects Ranking vs. Probability of No NBA Success',
                yaxis_title='Probability of No NBA Success'
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)

        if st.checkbox('Show all Draft Prospect Probabilities'):
            probs

        st.subheader("Create and Predict your own Draft Prospect with Support Vector Machine")
        rank = st.slider("Choose the Ranking of your player: ", min_value=1, max_value=60, value=30, step=1)
        age = st.slider("Choose the Age of your player: ", min_value=18.0, max_value=30.0, value=20.0, step=0.1)
        ft = st.slider("Choose the Free Throw Percentage of your player: ", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        stl = st.slider("Choose the Steal Percentage of your player: ", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        tpa = st.slider("Choose the Three Point Attempt Rate of your player: ", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

        user_prediction_data = [[rank, stl, ft, tpa, age]]
        if st.button('PREDICT'):
            a = clf.predict_proba(user_prediction_data)
            b = pd.DataFrame(a).round(4)
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

elif classifier!= 'Select Algorithm' and classifier =='Neural Network':
    X = df[['Ranking', 'ASTP', 'BLKP']].copy()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
    clf = MLPClassifier(solver='lbfgs', max_iter=500) 
    clf=clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    st.markdown('**Accuracy Score**')
    st.write(acc)
    auc = (metrics.roc_auc_score(y_test, clf.predict(X_test)))
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
    st.markdown('**Cross Validated Accuracy Score**')
    st.write(scores.mean())
    st.markdown('**AUC - ROC Score **')
    st.write(auc)
    if(auc > .50):
        st.image('https://raw.githubusercontent.com/neelganta/neel_project/master/NNCM.png')
    else:
        st.image('https://raw.githubusercontent.com/neelganta/neel_project/master/NN50CM.png')
    predict_new_dt = score[['Ranking', 'ASTP', 'BLKP']].copy()
    probs = clf.predict_proba(predict_new_dt)
    probs = pd.DataFrame(probs, columns=['Probability of No NBA Success', 'Probability of NBA Success'])
    probs['id'] = score['id']
    probs = probs[['id', 'Probability of No NBA Success', 'Probability of NBA Success']]
    ids = score['id'].tolist()
    ids.insert(0, 'Draft Prospects by ID') 
    st.subheader('Predicting Prospect Success with Neural Networks')
    selector = st.selectbox("Select a Draft Prospect's ID to predict their NBA success.", ids)     
    if selector != 'Draft Prospects by ID':
        a = score.loc[score['id'] == selector]
        a = a[['Ranking', 'ASTP', 'BLKP']] 
        a = clf.predict_proba(a)
        b = pd.DataFrame(a).round(4)
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
    
        st.subheader('Create your own Scatterplots of Draft Prospect Correlations')
        scatter = probs.set_index('id').join(score.set_index('id'))
        scatter = scatter.reset_index()
        xaxis = st.selectbox('Select which feature you want on your x-axis.', scatter.columns)
        yaxis = st.selectbox('Select which feature you want on your y-axis.', scatter.columns)

        if(xaxis != 'id' and yaxis !='id'):
            fig = px.scatter(scatter, x = xaxis, y =yaxis, hover_name = 'id', hover_data = [xaxis], trendline= 'ols' , color = yaxis)
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects ' + xaxis +' vs. ' + yaxis,
                yaxis_title= yaxis
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)
        else:
            fig = px.scatter(scatter, x = "Ranking", y ='Probability of No NBA Success', hover_name = 'id', hover_data = ['Probability of No NBA Success'],  trendline = 'ols', color = 'Probability of No NBA Success')
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                title_text='All Draft Prospects Ranking vs. Probability of No NBA Success',
                yaxis_title='Probability of No NBA Success'
            )

            fig.data[0].update(selectedpoints=3605,
                            selected=dict(marker=dict(color='red')),#color of selected points
                            unselected=dict(marker=dict(#color of unselected pts
                                            opacity=0.8)))
            st.plotly_chart(fig)

        if st.checkbox('Show all Draft Prospect Probabilities'):
            probs

        st.subheader("Create and Predict your own Draft Prospect with Neural Networks")
        rank = st.slider("Choose the Ranking of your player: ", min_value=1, max_value=60, value=30, step=1)
        ast = st.slider("Choose the Assist Percentage of your player: ", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
        blk = st.slider("Choose the Block Percentage of your player: ", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        user_prediction_data = [[rank, ast, blk]]

        if st.button('PREDICT'):
            a = clf.predict_proba(user_prediction_data)
            b = pd.DataFrame(a).round(4)
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
