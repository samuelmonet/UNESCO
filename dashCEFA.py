import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import pickle
import pydeck as pdk
import re
from collections import Counter
from PIL import Image

#import variables

#########################  a faire #########################################
# 
#
###########################################################################"


st.set_page_config(layout="wide")


#import des données
@st.cache
def load_data():
	data = pd.read_csv('viz.csv',sep='\t')
	data.drop([i for i in data if 'Unnamed' in i],axis=1,inplace=True)
	correl=pd.read_csv('graphs.csv',sep='\t')
	questions=pd.read_csv('questions.csv',sep='\t')
	questions.drop([i for i in questions.columns if 'Unnamed' in i],axis=1,inplace=True)
	quest=questions.iloc[3].to_dict()
	questions=questions.T
	sankey=questions[questions[1]=='sankey'].index.tolist()
	ridge=questions[questions[1]=='Viz'].index.tolist()
	codes=pd.read_csv('codes.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])


	return data,correl,quest,codes,sankey,ridge

data,correl,questions,codes,sankey,ridge=load_data()

#st.write(sankey)
#st.dataframe(correl)
#st.write(data.columns)
#st.write(correl.shape)

def sankey_graph(data,L,height=600,width=1600):
    """ sankey graph de data pour les catégories dans L dans l'ordre et 
    de hauter et longueur définie éventuellement"""
    
    nodes_colors=["blue","green","grey",'yellow',"coral",'darkviolet','saddlebrown','darkblue','brown']
    link_colors=["lightblue","limegreen","lightgrey","lightyellow","lightcoral",'plum','sandybrown','lightsteelblue','rosybrown']
    
    
    labels=[]
    source=[]
    target=[]
    
    for cat in L:
        lab=data[cat].unique().tolist()
        lab.sort()
        labels+=lab
    
    #st.write(labels)
    
    for i in range(len(data[L[0]].unique())): #j'itère sur mes premieres sources
    
        source+=[i for k in range(len(data[L[1]].unique()))] #j'envois sur ma catégorie 2
        index=len(data[L[0]].unique())
        target+=[k for k in range(index,len(data[L[1]].unique())+index)]
        
        for n in range(1,len(L)-1):
        
            source+=[index+k for k in range(len(data[L[n]].unique())) for j in range(len(data[L[n+1]].unique()))]
            index+=len(data[L[n]].unique())
            target+=[index+k for j in range(len(data[L[n]].unique())) for k in range(len(data[L[n+1]].unique()))]
       
    iteration=int(len(source)/len(data[L[0]].unique()))
    value_prov=[(int(i//iteration),source[i],target[i]) for i in range(len(source))]
    
    
    value=[]
    k=0
    position=[]
    for i in L:
        k+=len(data[i].unique())
        position.append(k)
    
   
    
    for triplet in value_prov:    
        k=0
        while triplet[1]>=position[k]:
            k+=1
        
        df=data[data[L[0]]==labels[triplet[0]]].copy()
        df=df[df[L[k]]==labels[triplet[1]]]
        #Je sélectionne ma première catégorie
        value.append(len(df[df[L[k+1]]==labels[triplet[2]]]))
        
    color_nodes=nodes_colors[:len(data[L[0]].unique())]+["black" for i in range(len(labels)-len(data[L[0]].unique()))]
    #st.write(color_nodes)
    color_links=[]
    for i in range(len(data[L[0]].unique())):
    	color_links+=[link_colors[i] for couleur in range(iteration)]
    #st.write(L,len(L),iteration)
    #st.write(color_links)
   
   
    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 30,
      line = dict(color = "black", width = 1),
      label = [i.upper() for i in labels],
      color=color_nodes
      )
      
    ,
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = target,
      value = value,
      color = color_links))])
    return fig


def count2(abscisse,ordonnée,dataf,legendtitle='',xaxis=''):
	
	dataf[ordonnée]=dataf[ordonnée].apply(lambda x:str(x))
	agg=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
	agg2=agg.T/agg.T.sum()
	agg2=agg2.T*100
	agg2=agg2.astype(int)
    
	if abscisse=='dam ':
		agg=agg.reindex(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
		agg2=agg2.reindex(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    
	x=agg.index
    
	#st.write(agg)
	#st.write(agg2)
	
	if ordonnée.split(' ')[0] in codes['Id'].values:
		#st.write('on est là')
		colors_code=codes[codes['Id']==ordonnée.split(' ')[0]].sort_values(['Coding'])
		labels=colors_code['label'].tolist()
		colors=colors_code['color'].tolist()
		fig = go.Figure()
		#st.write(labels,colors)
		#st.write(x)
		for i in range(len(labels)):
			if labels[i] in dataf[ordonnée].unique():
				#st.write(labels[i])
				#st.write(agg.columns)
				#st.write(agg[(abscisse,str(labels[i]))])
				fig.add_trace(go.Bar(x=x, y=agg[(abscisse,str(labels[i]))], name=str(labels[i]),\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,str(labels[i]))],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        
	else:
		fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green',customdata=agg2.iloc[:,0],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
		for i in range(len(agg.columns)-1):
			fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1],customdata=agg2.iloc[:,i+1],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
    
	fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Persons','title_font':{'size':18}})
	fig.update_layout(legend_title=legendtitle,legend=dict(orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    #fig.update_layout(title_text=title)
    
	return fig

def pourcent2(abscisse,ordonnée,dataf,legendtitle='',xaxis=''):
    
	agg2=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
	agg=agg2.T/agg2.T.sum()
	agg=agg.T.round(2)*100
    
    
	if abscisse=='dam ':
		agg=agg.reindex(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
		agg2=agg2.reindex(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    
	x=agg2.index
    
	if ordonnée.split(' ')[0] in codes['Id'].values:
		colors_code=codes[codes['Id']==ordonnée.split(' ')[0]].sort_values(['Coding'])
		labels=colors_code['label'].tolist()
		colors=colors_code['color'].tolist()
		fig = go.Figure()
        
		for i in range(len(labels)):
			if labels[i] in dataf[ordonnée].unique():
				fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
        
	else:
        #st.write(agg)
        #st.write(agg2)
		fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green',customdata=agg2.iloc[:,0],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
		for i in range(len(agg.columns)-1):
			fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1],customdata=agg2.iloc[:,i+1],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
    
	fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Percentages','title_font':{'size':18}})
	fig.update_layout(legend_title=legendtitle,legend=dict(orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    #fig.update_layout(title_text=title)    
	return fig


img1 = Image.open("logoAxiom.png")
img2 = Image.open("logoUNESCO.png")
#img3 = Image.open("logoAVSI.png")

def main():	
	
	
	
	#st.write(codes)
	
	st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")
	
	title1, title3 = st.columns([9,2])

	
	topic = st.sidebar.radio('What do you want to do ?',('Display machine learning results','Display correlations',\
		'Display Sankey Graphs','Display Wordclouds'))
	title3.image(img2)
		
	
	if topic=='Display machine learning results':
		
		title1.title('Machine learning results on predictive model trained on : ')
		title1.title('Question: Can you say that your agricultural productivity has increased as result of the project')
		
		
		st.title('')
		st.markdown("""---""")	
		st.subheader('Note:')
		st.write('A machine learning model has been run on the question related to impression of increase of productivity, the objective of this was to identify specificaly for these question which are the parameters that influence the most the increase of productivity. The models are run in order to try to predict as precisely as possible the feeling that the respondents expressed in their response to this question. The figures below shows which parameters have a greater impact in the prediction of the model than a normal random aspect (following a statistic normal law)')
		st.write('')
		st.write('Each line of the graph represents one feature of the survey that is important to predict the response to the question.')
		st.write('Each point on the right of the feature name represents one person of the survey. A red point represent a high value to the specific feature and a blue point a low value (a purple one a value inbetween).')
		st.write('SHAP value: When a point is on the right side, it means that it contributed to a better note while on the left side, this specific caracter of the person reduced the final result of the prediction. In this case it is 1 for productivity increase and 0 for no increase.')
		st.write('')
		st.write('The coding for the responses is indicated under the graph and the interpretation of the graphs is written below.')
		st.markdown("""---""")	
				
		temp = Image.open('shap.png')
		image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.image(image, use_column_width = True)
		
		a, b = st.columns([1,1])
		a.caption('What assistance did you receive from CEFA? [Agricultural support] \n Received 1 - Did not receive : 0')
		a.caption('What assistance did you receive from CEFA? [Training] Received 1 - Did not receive : 0')
		a.caption('Do you have a have a sick person in the household who has NOT received medical care due to money/costs? Yes:1 No:0')
		a.caption('What crops did you grow on this plot? [Cowpea]  Grows Cowpea : 1 - Does not : 1')
		a.caption("Did you receive any agricultural support from the Emergency Project by AICS? Yes:1 No:0")
		b.caption("Have you received any of the following agricultural training in the last 12 months from CEFA? [Use of agricultural inputs] Yes:1 No:0")
		b.caption("Did you or any other member of your household (available) use a latrine/toilet the last time you defecated? Yes:1 No:0")
		b.caption("What is your current MAIN source of water for drinking, cooking, and hygiene in your household during the DRY season?[Protected Dug Well] Protected dug well: 1 Not protected dug well: 0")
		
		st.write('We can see that the main parameter for feeling agricultural productivity has increased as result of the project are the fact to have received agricultural support (including trainings). This seems to show the effectiveness of these two activities')
		st.write('We see also that people who grew cowpea are also more likely to have seen their productivity increased.')
		st.write('This is also the case of people whose main source of water is protected dugwells and less likely the case in villages with many boreholes.')
		st.write('Larger family are also mre likely to have seen their producivity increased but this might be the fact that the project targetted mainly large households.')
		st.write('Finaly we also see that having a people sick in the household for whom people can not afford treatment tends to decresase the probability to see an increase in the productivity')
		
		
		st.markdown("""---""")	
	
	
	
				
		
	elif topic=='Display correlations':	
		
		title1.title('Main correlations uncovered from the database')
		st.write('Note: Correlation does not mean causation. This is not because 2 features are correlated that one is the cause of the other. So conclusion have to be made with care.')
		continues=pickle.load( open( "cont_feat.p", "rb" ) )
		cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
		
				
		quests=correl[correl['variable_x'].fillna('').apply(lambda x: True if 'region' not in x else False)]
		
		#st.write(quest)
		#st.write(codes)
		#st.write(cat_cols)
		
		#st.write(data['assistancetype'].value_counts())
		
			
							
		for absc in quests['variable_x'].unique():
			
			k=0
					
			
			quest=quests[quests['variable_x']==absc]
			
			#st.write(quest)
			
			if len(quest)>1 or 'bar' in quest['graphtype'].unique():
				col1,col2=st.columns([1,1])
			
			for i in range(len(quest)):
				
				
				
				#st.write(quest.iloc[i]['variable_x']+'##')
				#st.write('in cat cols: ',quest.iloc[i]['variable_x'] in cat_cols)
				
				canal=['canalworkingwell','monthscanaldry','increasedarea']
				if absc in canal or quest.iloc[i]['variable_y'] in canal:
					#st.write('coucou')
					datas=data[data['canalrehab']=='Yes'].copy()
				else:
					datas=data.copy()
			
				if quest.iloc[i]['variable_x'] in cat_cols or quest.iloc[i]['variable_y'] in cat_cols:
					
					if quest.iloc[i]['variable_x'] in cat_cols:
						cat,autre=quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']
					else:
						cat,autre=quest.iloc[i]['variable_y'],quest.iloc[i]['variable_x']
					#st.write('cat: ',cat,' et autre: ',autre)
						
					df=pd.DataFrame(columns=[cat,autre])
					
					catcols=[j for j in datas.columns if cat in j]
					cats=[' '.join(i.split(' ')[1:])[:57] for i in catcols]
				
					for n in range(len(catcols)):
						ds=datas[[catcols[n],autre]].copy()
						ds=ds[ds[catcols[n]].isin(['Yes',1])]
						ds[catcols[n]]=ds[catcols[n]].apply(lambda x: cats[n])
						ds.columns=[cat,autre]
						df=df.append(ds)
					df['persons']=np.ones(len(df))		
					#st.write(df)		
					
					#st.write(quest.iloc[i]['graphtype'])
						
									
				else:	
					df=datas[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']]].copy()
					df['persons']=np.ones(len(df))
				
				if quest.iloc[i]['graphtype']=='sunburst':
					st.subheader(quest.iloc[i]['title'])
					fig = px.sunburst(df.fillna(''), path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], 	values='persons',color=quest.iloc[i]['variable_y'])
					#fig.update_layout(title_text=quest.iloc[i]['variable_x'] + ' and ' +quest.iloc[i]['variable_y'],font=dict(size=20))
					st.plotly_chart(fig,size=1000)
					
					
					
				
				elif quest.iloc[i]['graphtype']=='treemap':
					
					st.subheader(quest.iloc[i]['title'])
					fig=px.treemap(df, path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons',color=quest.iloc[i]['variable_y'])
					#fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					
					st.plotly_chart(fig,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					
				
					
				elif quest.iloc[i]['graphtype']=='violin':
					
					fig = go.Figure()
				
					if quest.iloc[i]['variable_x'].split(' ')[0] in codes['Id'].unique():
						categs = codes[codes['Id']==quest.iloc[i]['variable_x'].split(' ')[0]].sort_values(by='coding')['label'].tolist()				
					
					else:
						categs = df[quest.iloc[i]['variable_x']].unique()
					
					for categ in categs:
					    fig.add_trace(go.Violin(x=df[quest.iloc[i]['variable_x']][df[quest.iloc[i]['variable_x']] == categ],
		                            		y=df[quest.iloc[i]['variable_y']][df[quest.iloc[i]['variable_x']] == categ],
		                            		name=categ,
		                            		box_visible=True,
        	                   			meanline_visible=True,points="all",))
					
					
					fig.update_layout(showlegend=False)
					fig.update_yaxes(range=[-0.1, df[quest.iloc[i]['variable_y']].max()+1])
					fig.update_layout(yaxis={'title':quest.iloc[i]['ytitle'],'title_font':{'size':18}})	
					k+=1
					
					
					
					if len(quest[quest['graphtype']=='violin'])>2:
						
						if k%2==1:
							col1.subheader(quest.iloc[i]['title'])
							col1.plotly_chart(fig,use_container_width=True)
							
						else:
							col2.subheader(quest.iloc[i]['title'])
							col2.plotly_chart(fig,use_container_width=True)
							
					else:
						st.subheader(quest.iloc[i]['title'])
						st.plotly_chart(fig,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					
									
				elif quest.iloc[i]['graphtype']=='bar':
					
					#st.write(df[quest.iloc[i]['variable_y']].dtype)
					
					
					st.subheader(quest.iloc[i]['title'])
				
					col1,col2=st.columns([1,1])

					fig1=count2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
					df,legendtitle=quest.iloc[i]['legendtitle'],xaxis=quest.iloc[i]['xtitle'])
					
					col1.plotly_chart(fig1,use_container_width=True)
						
					fig2=pourcent2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
					df,legendtitle=quest.iloc[i]['legendtitle'],xaxis=quest.iloc[i]['xtitle'])
					#fig2.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=True,xaxis_tickangle=45)
					col2.plotly_chart(fig2,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					#st.write(df)
				
				st.markdown("""---""")		
	
				
	
		
	elif topic=='Display Sankey Graphs':
	
		title1.title('Visuals for questions related to cultures (Some of questions B3 to B17)')
		st.title('')
		
		
		crops=[i for i in data if i[0]=='B' and i[:3] not in ['B1_','B20','B19']]
		#st.write(sankey)	
		#st.write(crops)
		
		data_all=data[sankey+['productivity_increased']].copy()
		
		#st.write()
		
		cowpea=data_all[['productivity_increased']+[i for i in crops if 'cowpea' in i.lower()]].copy()
		sorghum=data_all[['productivity_increased']+[i for i in crops if 'sorghum' in i.lower()]].copy()
		melon=data_all[['productivity_increased']+[i for i in crops if 'melon' in i.lower()]].copy()
		maize=data_all[['productivity_increased']+[i for i in crops if 'maize' in i.lower()]].copy()
		cabbage=data_all[['productivity_increased']+[i for i in crops if 'cabbage' in i.lower()]].copy()
		tomatoes=data_all[['productivity_increased']+[i for i in crops if 'tomatoes' in i.lower()]].copy()
		
		
		#st.write(cowpea)
		
		colonnes=['Seeds Planted','Type of seeds','Origin of seeds','Did you have adequate/enough seed?',\
          'Did any pests or diseases attack your crops?','Did you practice integrated pest management for this crop?',\
          'Did you apply any of the following good agricultural practices for this crop?','Did you practice any irrigation for this crop?']
		for i in [cowpea,sorghum,melon,maize,cabbage,tomatoes]:
    			i.columns=['productivity_increased']+colonnes
		cowpea=cowpea[cowpea['Seeds Planted']=='Yes']
		sorghum=sorghum[sorghum['Seeds Planted']=='Yes']
		melon=melon[melon['Seeds Planted']=='Yes']
		maize=maize[maize['Seeds Planted']=='Yes']
		cabbage=cabbage[cabbage['Seeds Planted']=='Yes']
		tomatoes=tomatoes[tomatoes['Seeds Planted']=='Yes']
		
		
		cowpea['Seeds Planted']=cowpea['Seeds Planted'].apply(lambda x: 'Cowpea')
		sorghum['Seeds Planted']=sorghum['Seeds Planted'].apply(lambda x: 'Sorghum')
		melon['Seeds Planted']=melon['Seeds Planted'].apply(lambda x: 'Melon')
		maize['Seeds Planted']=maize['Seeds Planted'].apply(lambda x: 'Maize')
		cabbage['Seeds Planted']=cabbage['Seeds Planted'].apply(lambda x: 'Cabbage')
		tomatoes['Seeds Planted']=tomatoes['Seeds Planted'].apply(lambda x: 'Tomatoes')
		
		
		sank=pd.DataFrame(columns=['productivity_increased']+colonnes)
		for i in [cowpea,sorghum,melon,maize,cabbage,tomatoes]:
		    sank=sank.append(i)
		sank['ones']=np.ones(len(sank))
		
		st.title('Some examples')
		
		st.markdown("""---""")
		st.write('Seeds Planted - Did you practice any irrigation for this crop? - '+questions['productivity_increased'])
		fig=sankey_graph(sank,['Seeds Planted','Did you practice any irrigation for this crop?','productivity_increased'],height=600,width=1500)
		fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig,use_container_width=True)
		st.write('We can see that those who practice irrigation have seen more increase in their productivity')
		
		st.markdown("""---""")
		
		st.write('Seeds Planted - Origin of seeds - Did any pests or diseases attack your crops?')
		fig=sankey_graph(sank,['Seeds Planted','Origin of seeds','Did any pests or diseases attack your crops?'],height=600,width=1500)
		fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig,use_container_width=True)
		st.write('Surprisingly, we can see that seeds from NGO seem to have been less resilient to pest attack than saved seeds or seeds from government and local market. Also Cowpea and Melon seem to be less vulnerable to pest and diseases.')
		
		st.markdown("""---""")
		
		if st.checkbox('Design my own Sankey Graph'):
			
			st.markdown("""---""")
			feats=st.multiselect('Select features you want to see in the order you want them to appear', [questions['productivity_increased']]+colonnes)
			
			if len(feats)>=2:
				st.write(' - '.join(feats))
				a=False
				for i in feats:
					if i in colonnes:
						a=True
				if a:
					df=sank.copy()
				else:
					df=data_all
				
				features=[]
				for i in feats:
					if i in colonnes:
						features.append(i)
					else:
						features.append([n for n in questions if questions[n]==i][0])
				
				#st.write(features)
				
				fig3=sankey_graph(df,features,height=600,width=1500)
				fig3.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
				st.plotly_chart(fig3,use_container_width=True)
		
		
	
	
	

    
 
if __name__== '__main__':
    main()




    
