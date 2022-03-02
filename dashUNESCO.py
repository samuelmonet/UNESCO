import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pickle
import re
from collections import Counter
from PIL import Image
from joypy import joyplot

st.set_page_config(layout="wide")


@st.cache
def load_data():
	data = pd.read_csv('viz2.csv', sep='\t')
	data.drop([i for i in data if 'Unnamed' in i], axis=1, inplace=True)
	continues = pickle.load(open("cont_feat.p", "rb"))
	for i in continues:
		data[i] = data[i].fillna(0).apply(lambda x : float(x))
	data['Village_clean'] = data['Village_clean'].apply(lambda x: 'Al-Samoud' if
														x == 'Al-Samoud neighborhood, Alshhid Badr unit' else x)
	data['cashspend_num'] = data['cashspend_num'].astype('str')
	correl = pd.read_csv('graphs.csv',sep='\t')
	questions = pd.read_csv('questions.csv',sep='\t')
	questions.drop([i for i in questions.columns if 'Unnamed' in i], axis=1, inplace=True)
	quest = questions.iloc[4].to_dict()
	cfw = pd.read_csv('datancfw.csv',sep='\t')
	cfw['Village_clean'] = cfw['Village_clean'].apply(
		lambda x: 'Al-Samoud' if x == 'Al-Samoud neighborhood, Alshhid Badr unit' else x)
	codes = pd.read_csv('codes.csv', index_col=None, sep='\t').dropna(how='any', subset=['color'])
	
	return data, correl, quest, codes, cfw


data, correl, questions, codes, cfw = load_data()


def sankey_graph(data, L, height=600,width=1600):
	""" sankey graph de data pour les catégories dans L dans l'ordre et  de hauter et longueur définie éventuellement"""
	nodes_colors = ["blue", "green", "grey", 'yellow', "coral", 'darkviolet', 'saddlebrown', 'darkblue', 'brown']
	link_colors = ["lightblue", "limegreen", "lightgrey", "lightyellow", "lightcoral", 'plum', 'sandybrown', 'lightsteelblue', 'rosybrown']
	labels = []
	source = []
	target = []
	for cat in L:
		lab = data[cat].unique().tolist()
		lab.sort()
		labels += lab
	for i in range(len(data[L[0]].unique())):  # j'itère sur mes premieres sources
		source+=[i for k in range(len(data[L[1]].unique()))]  # j'envois sur ma catégorie 2
		index=len(data[L[0]].unique())
		target+=[k for k in range(index,len(data[L[1]].unique())+index)]
		for n in range(1, len(L)-1):
			source += [index+k for k in range(len(data[L[n]].unique())) for j in range(len(data[L[n+1]].unique()))]
			index += len(data[L[n]].unique())
			target += [index+k for j in range(len(data[L[n]].unique())) for k in range(len(data[L[n+1]].unique()))]
	iteration = int(len(source)/len(data[L[0]].unique()))
	value_prov = [(int(i//iteration), source[i], target[i]) for i in range(len(source))]
	value = []
	k = 0
	position = []
	for i in L:
		k += len(data[i].unique())
		position.append(k)
	for triplet in value_prov:
		k = 0
		while triplet[1] >= position[k]:
			k += 1
		df = data[data[L[0]] == labels[triplet[0]]].copy()
		df = df[df[L[k]] == labels[triplet[1]]]
		value.append(len(df[df[L[k+1]] == labels[triplet[2]]]))
	color_nodes=nodes_colors[:len(data[L[0]].unique())]+["black" for i in range(len(labels)-len(data[L[0]].unique()))]
	color_links=[]
	for i in range(len(data[L[0]].unique())):
		color_links += [link_colors[i] for couleur in range(iteration)]
	fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=30, line=dict(color="black", width=1),
											  label=[i.upper() for i in labels], color=color_nodes),
									Link=dict(source=source, target=target, value=value, color=color_links))])
	return fig


def count2(abscisse, ordonnee, dataf, legendtitle='', xaxis=''):
	
	dataf[ordonnee] = dataf[ordonnee].apply(lambda x : str(x))
	agg = dataf[[abscisse, ordonnee]].groupby(by=[abscisse, ordonnee]).aggregate({abscisse : 'count'}).unstack().fillna(0)
	agg2 = agg.T/agg.T.sum()
	agg2 = agg2.T*100
	agg2 = agg2.astype(int)
	if abscisse == 'Village_clean':
		agg = agg.reindex(['Bit Boos', "Old Sana'a", "Enma'a", "Alkatea'a", "Hada'a", 'AlGamea', "Alomall neighborhood",
						   'Al-Samoud'])
		agg2 = agg2.reindex(['Bit Boos', "Old Sana'a", "Enma'a", "Alkatea'a", "Hada'a", 'AlGamea', "Alomall neighborhood",
							 'Al-Samoud'])

	x = agg.index

	if ordonnee.split(' ')[0] in codes['list name'].values:
		#st.write('on est là')
		colors_code = codes[codes['list name'] == ordonnee.split(' ')[0]].sort_values(['coding'])
		labels = colors_code['label'].tolist()
		colors = colors_code['color'].tolist()
		fig = go.Figure()
		for i in range(len(labels)):
			if labels[i] in dataf[ordonnee].unique():
				fig.add_trace(go.Bar(x=x, y=agg[(abscisse,str(labels[i]))], name=str(labels[i]),
									marker_color=colors[i].lower(),customdata=agg2[(abscisse,str(labels[i]))],textposition="inside",
									texttemplate="%{customdata} %",textfont_color="black"))
	else:
		fig = go.Figure(go.Bar(x=x, y=agg.iloc[:, 0], name=agg.columns.tolist()[0][1], marker_color='green',
								customdata=agg2.iloc[:, 0], textposition="inside",
							   texttemplate="%{customdata} %", textfont_color="black"))
		for i in range(len(agg.columns)-1):
			fig.add_trace(go.Bar(x=x, y=agg.iloc[:, i+1], name=agg.columns.tolist()[i+1][1], customdata=agg2.iloc[:,i+1],
								 textposition="inside", texttemplate="%{customdata} %", textfont_color="black"))
	fig.update_layout(barmode='relative', xaxis={'title':xaxis,'title_font':{'size':18}},
					  yaxis={'title':'Persons','title_font':{'size': 18}}
					)
	fig.update_layout(legend_title=legendtitle,legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",
														   x=1.01, font=dict(size=18), title=dict(font=dict(size=18)))
					)
	return fig


def pourcent2(abscisse, ordonnee, dataf, legendtitle='', xaxis=''):
	agg2 = dataf[[abscisse, ordonnee]].groupby(by=[abscisse, ordonnee]).aggregate({abscisse : 'count'}).unstack().fillna(0)
	agg = agg2.T/agg2.T.sum()
	agg = agg.T.round(2)*100
	if abscisse == 'Village_clean':
		agg = agg.reindex(
			['Bit Boos', "Old Sana'a", "Enma'a", "Alkatea'a", "Hada'a", 'AlGamea', "Alomall neighborhood", 'Al-Samoud'])
		agg2 = agg2.reindex(
			['Bit Boos', "Old Sana'a", "Enma'a", "Alkatea'a", "Hada'a", 'AlGamea', "Alomall neighborhood", 'Al-Samoud'])
	x = agg.index
	x = agg2.index
	if ordonnee.split(' ')[0] in codes['list name'].values:
		colors_code = codes[codes['list name']==ordonnee.split(' ')[0]].sort_values(['coding'])
		labels = colors_code['label'].tolist()
		colors = colors_code['color'].tolist()
		fig = go.Figure()
		for i in range(len(labels)):
			if labels[i] in dataf[ordonnee].unique():
				fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i], marker_color=colors[i].lower(),
									 customdata=agg2[(abscisse,labels[i])], textposition="inside",
									 texttemplate="%{customdata} persons",textfont_color="black")
				)
	else:
		fig = go.Figure(go.Bar(x=x, y=agg.iloc[:, 0], name=agg.columns.tolist()[0][1], marker_color='green',
							customdata=agg2.iloc[:, 0], textposition="inside", texttemplate="%{customdata} persons",
							textfont_color="black")
		)
		for i in range(len(agg.columns)-1):
			fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1], customdata=agg2.iloc[:, i+1],
								textposition="inside", texttemplate="%{customdata} persons", textfont_color="black")
				)
	fig.update_layout(barmode='relative', xaxis={'title':xaxis,'title_font':{'size':18}},
					  yaxis={'title':'Percentages','title_font':{'size':18}}
					  )
	fig.update_layout(legend_title=legendtitle,legend=dict(orientation='h', yanchor="bottom", y=1.02, xanchor="right",
														   x=1.01,font=dict(size=18),title=dict(font=dict(size=18)))
					  )
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

	
	topic = st.sidebar.radio('What do you want to do ?',('Display machine learning results', 'Display correlations',
														 'Display Other visuals', 'Analyze Wordclouds'))
	title3.image(img2)

	if topic == 'Display machine learning results':
		
		title1.title('Machine learning results on predictive model trained on Questions:')
		title1.title('- How long the cash by the CFW project received lasted?')
		title1.title('- How long were the effects of the cash you received from the cash for work project?')
		st.title('')
		st.markdown("""---""")	
		st.subheader('Note:')
		st.write('A machine learning model has been run on the question related to the lasting effects of the project, '
				 'the objective of this was to identify specificaly for these question which are the parameters that'
				 ' influenced it the most. The models are run in order to try to predict as precisely as possible '
				 'the lasting effects that the respondents expressed in their response to this question. '
				 'The figures below shows which parameters have a greater impact in the prediction '
				 'of the model than a normal random aspect (following a statistic normal law)')
		st.write('')
		st.write('Each line of the graph represents one feature of the survey that is important to predict '
				 'the response to the question.')
		st.write('Each point on the right of the feature name represents one person of the survey. '
				 'A red point represent a high value to the specific feature and a blue point a '
				 'low value (a purple one a value inbetween).')
		st.write('SHAP value: When a point is on the right side, it means that it contributed to a '
				 'longer effect note while on the left side, this specific caracter of the person '
				 'reduced the final result of the prediction.')
		st.write('')
		st.write('The coding for the responses is indicated under the graph and '
				 'the interpretation of the graphs is written below.')
		st.markdown("""---""")	

		st.title('How long the cash by the CFW project received lasted?')
		temp = Image.open('shap1.png')
		image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.image(image, use_column_width = True)
		
		st.caption('Do you know how your daily salary was decided: Yes : 1 - No : 0')
		st.caption('The cash received allowed you to increase expenditures for: Meat - Responded Meat : 1 - Did not mention Meat : 0')
		st.caption("What are usually the most difficult months for your household:Febrauary - Responded February : 1 - Did not mention February : 0")

		st.write('One of the main factor for having a long effect of the cash received from the project seems to be '
				 'related to the way people used their cash:')
		st.write('- People who used it mainly for Cowpea, Kerozene, Soap, Sugar and Meat tends to have had it lasting longer')
		st.write('- On the other hand those who used it mainly on Tea leaves, Fruits and Qhat have had it lasting shorter')
		st.write('People who knew how the salary was decided also tend to have had their cash lasting longer. This coud be related to the villages'
				 'where the project was implemented as the 2 features are quite correlated as we can see on correlations.')
		st.write('When a high percentage of adults member of the household participated in the CFW activities this tends also to increase the time the cash lasted.')
		st.write('Participation of women in the decision on how to use the cash seem also to be an important factor for having the cash lasting longer')
		st.write('Finaly, it seems that February is often among the most difficult month for those who had their cash lasting longer.')

		st.markdown("""---""")

		st.title('How long were the effects of the cash you received from the cash for work project?')
		temp = Image.open('shap2.png')
		image = Image.new("RGBA", temp.size, "WHITE")  # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.image(image, use_column_width=True)

		st.write('Here again, one of the main factor for lasting effects is the use that was made of cash:')
		st.write('- People who invested in cash, health and clothes saw longer effects')
		st.write("- People who used it mainly for Qhat had much shorter effects")
		st.write(
			'On this aspect the role of women seems to be particularly important. When women participate in the decision making on'
			'how the cash is used, the effect of the cash tends to be longer.')
		st.write('This is also the case when women participate directly to CFW activities.')
		st.write('Finaly, it seems that November is often among the most difficult month for those who have had shorter effect.')
		st.markdown("""---""")

	elif topic == 'Display correlations':
		st.title('Main correlations uncovered from the database:	')
		st.write('Note: Correlation does not mean causation. This is not because 2 features are correlated that one is '
				 'the cause of the other. So conclusion have to be made with care.')
		continues = pickle.load( open( "cont_feat.p", "rb" ) )
		cat_cols = pickle.load( open( "cat_cols.p", "rb" ) )
		st.markdown("""---""")
		quests = correl[correl['variable_x'].fillna('').apply(lambda x : True if 'region' not in x else False)]
		k = 0
		for absc in quests['variable_x'].unique():
			quest = quests[quests['variable_x'] == absc]
			if len(quest)>1 or 'bar' in quest['graphtype'].unique():
				col1,col2=st.columns([1,1])
			for i in range(len(quest)):
				if absc == 'CFW':
					datas = cfw.copy()
				else:
					datas = data.copy()
				if quest.iloc[i]['variable_x'] in cat_cols or quest.iloc[i]['variable_y'] in cat_cols:
					if quest.iloc[i]['variable_x'] in cat_cols:
						cat, autre = quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']
					else:
						cat, autre = quest.iloc[i]['variable_y'],quest.iloc[i]['variable_x']
					df = pd.DataFrame(columns=[cat,autre])
					catcols = [j for j in datas.columns if cat in j]
					cats = [' '.join(i.split(' ')[1:]) for i in catcols]
					for n in range(len(catcols)):
						ds = datas[[catcols[n], autre]].copy()
						ds = ds[ds[catcols[n]].isin(['Yes', 1])]
						ds[catcols[n]] = ds[catcols[n]].apply(lambda x : cats[n])
						ds.columns = [cat, autre]
						df = df.append(ds)
					df['persons'] = np.ones(len(df))
				else:
					df = datas[[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']]].copy()
					df['persons'] = np.ones(len(df))
				#st.write(df)
				if quest.iloc[i]['graphtype'] == 'treemap':
					st.subheader(quest.iloc[i]['title'])
					#fig=go.Figure()
					#fig.add_trace(go.Treemap(branchvalues='total',labels=data[quest.iloc[i]['variable_x']],parents=data[quest.iloc[i]['variable_y']],
					#			  root_color="lightgrey",textinfo="label+value"))
					fig = px.treemap(df, path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']],
								   values='persons',color=quest.iloc[i]['variable_y'])
					st.plotly_chart(fig, use_container_width=True)
					st.write(quest.iloc[i]['description'])

				elif quest.iloc[i]['graphtype'] == 'violin':
					fig = go.Figure()
					if quest.iloc[i]['variable_x'].split(' ')[0] in codes['list name'].unique():
						categs = codes[codes['Id'] == quest.iloc[i]['variable_x'].split(' ')[0]].sort_values(by='coding')['label'].tolist()
					elif quest.iloc[i]['variable_x'] == 'Village_clean':
						categs = ['Bit Boos',"Old Sana'a", "Enma'a", "Alkatea'a", "Hada'a", 'AlGamea', "Alomall neighborhood", 'Al-Samoud']
					elif quest.iloc[i]['variable_x'] == 'cashspend_num':
						categs = ['0', '1', '2', '3', '4']
					elif quest.iloc[i]['variable_x'] == 'educ_mean':
						categs = ['Koranic school','Some primary school','Completed primary school',
								'Some secondary school','Completed secondary school','University']

					else:
						categs = df[quest.iloc[i]['variable_x']].unique()

					for categ in categs:
						fig.add_trace(go.Violin(x=df[quest.iloc[i]['variable_x']][df[quest.iloc[i]['variable_x']] == str(categ)],
										y=df[quest.iloc[i]['variable_y']][df[quest.iloc[i]['variable_x']] == str(categ)],
										name=categ,
										box_visible=True,
										meanline_visible=True,points="all",))

					fig.update_layout(showlegend=False)
					fig.update_yaxes(range=[-0.1, df[quest.iloc[i]['variable_y']].max()+1])
					fig.update_layout(yaxis={'title' : quest.iloc[i]['ytitle'], 'title_font' : {'size' : 18}})

					st.subheader(quest.iloc[i]['title'])

					if quest.iloc[i]['nonCFW'] == 'X':
						k += 1
						if st.checkbox(str(k)+' - Show also not CFW Beneficiaries'):
							df1 = cfw[cfw['CFW'] == 'CFW Beneficiary'].copy()
							df2 = cfw[cfw['CFW'] != 'CFW Beneficiary'].copy()
							col1, col2 = st.columns([1, 1])
							col1.subheader('CFW beneficiaries')
							col2.subheader('Non CFW beneficiaries')
							fig1, fig2 = go.Figure(), go.Figure()
							for categ in categs:
								fig1.add_trace(go.Violin(
									x=df1[quest.iloc[i]['variable_x']][df1[quest.iloc[i]['variable_x']] == str(categ)],
									y=df1[quest.iloc[i]['variable_y']][df1[quest.iloc[i]['variable_x']] == str(categ)],
									name=categ,
									box_visible=True,
									meanline_visible=True, points="all", ))
								fig2.add_trace(go.Violin(
									x=df2[quest.iloc[i]['variable_x']][df2[quest.iloc[i]['variable_x']] == str(categ)],
									y=df2[quest.iloc[i]['variable_y']][df2[quest.iloc[i]['variable_x']] == str(categ)],
									name=categ,
									box_visible=True,
									meanline_visible=True, points="all", ))
							fig1.update_layout(showlegend=False)
							fig1.update_yaxes(range=[-0.1, df1[quest.iloc[i]['variable_y']].max() + 1])
							fig1.update_layout(yaxis={'title': quest.iloc[i]['ytitle'], 'title_font': {'size': 18}})
							fig2.update_layout(showlegend=False)
							fig2.update_yaxes(range=[-0.1, df2[quest.iloc[i]['variable_y']].max() + 1])
							fig2.update_layout(yaxis={'title': quest.iloc[i]['ytitle'], 'title_font': {'size': 18}})
							col1.plotly_chart(fig1, use_container_width=True)
							col2.plotly_chart(fig2, use_container_width=True)
							st.write(quest.iloc[i]['Description2'])  # Mettre une autre description
						else:
							st.plotly_chart(fig, use_container_width=True)
							st.write(quest.iloc[i]['description'])
					else:
						st.plotly_chart(fig,use_container_width=True)
						st.write(quest.iloc[i]['description'])

				elif quest.iloc[i]['graphtype'] == 'bar':
					#st.write(df[quest.iloc[i]['variable_y']].dtype)
					st.subheader(quest.iloc[i]['title'])
					if quest.iloc[i]['nonCFW'] == 'X':
						st.subheader('CFW Beneficiaries')
					col1, col2 = st.columns([1, 1])
					fig1 = count2(quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y'],
					df, legendtitle=quest.iloc[i]['legendtitle'], xaxis=quest.iloc[i]['xtitle'])
					col1.plotly_chart(fig1, use_container_width=True)
					fig2 = pourcent2(quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y'],
					df, legendtitle=quest.iloc[i]['legendtitle'], xaxis=quest.iloc[i]['xtitle'])
					col2.plotly_chart(fig2, use_container_width=True)
					st.write(quest.iloc[i]['description'])
					if quest.iloc[i]['nonCFW'] == 'X':
						k += 1
						if st.checkbox(str(k) + ' - Show also not CFW Beneficiaries'):
							st.subheader('Non CFW Beneficiaries')
							col1, col2 = st.columns([1, 1])
							fig1 = count2(quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y'],
										cfw[cfw['CFW'] != 'CFW Beneficiary'], legendtitle=quest.iloc[i]['legendtitle'],
										xaxis=quest.iloc[i]['xtitle'])
							col1.plotly_chart(fig1, use_container_width=True)
							fig2 = pourcent2(quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y'],
											 cfw[cfw['CFW'] != 'CFW Beneficiary'], legendtitle=quest.iloc[i]['legendtitle'], xaxis=quest.iloc[i]['xtitle'])
							# fig2.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=True,xaxis_tickangle=45)
							col2.plotly_chart(fig2, use_container_width=True)
							st.write(quest.iloc[i]['Description2'])
						#st.write(df)

				st.markdown("""---""")

	elif topic == 'Display Other visuals':
		st.markdown("""---""")
		st.title('Rank how have you used the cash received?')
		st.write('You can move the box with the mouse and increase the size of the plot on the right corner if you need')
		sank = data[['use1', 'use2', 'use3']].copy()
		L = ['Wheat flour', 'Rice', 'Qhat', 'Tools', 'Clothes', 'Health', 'Meat', 'Education']
		sank['use1'] = sank['use1'].apply(lambda x : x if x in L else 'Other')
		st.write('First main usage - Second main usage - Third main usage')
		fig = sankey_graph(sank, ['use1', 'use2', 'use3'], height=900, width=1500)
		fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey')
		st.plotly_chart(fig, use_container_width=True, height=900, t_margin=0, b_margin=0)
		st.markdown("""---""")
		st.title('Distribution of percentages of use of the cash received per category')
		df = pd.DataFrame(columns=['usage', 'percentage'])
		for i in range(1, 27):
			dftemp = pd.DataFrame(columns=['usage', 'village', 'percentage'])
			dftemp['usage'] = np.ones(len(data))
			dftemp['usage'] = dftemp['usage'].apply(lambda x: questions['usage'+str(i)])
			dftemp['percentage'] = data['usage' + str(i)]
			dftemp['village'] = data['Village_clean']
			df = df.append(dftemp)
		df = df[df['percentage'] > 0]
		#st.write(df)
		fig, ax = joyplot(data=df, by='usage')
		st.pyplot(fig)

	elif topic == 'Analyze Wordclouds':
		text = pickle.load(open("text.p", "rb"))
		continues = pickle.load(open("cont_feat.p", "rb"))
		to_drop = pickle.load(open("drop.p", "rb"))
		quest_list = ['Reason why the selected month is difficult',
					'Do you know why you were selected to participate in this project ?',
					'Which type of training would you like to receive?',
					'Explain which changes occurred in your life thanks to the CFW',
					'Explain what happened to others youth in similar needs of CFW who did not access the program',
					'Enter livelihood category',
					'Which skills did you learn ?',
					'What are you doing NOW in terms of incomes generation?',
					'What would stop you to do more of this work ?',
					'Most important things you learnt during this cash for work project in terms of Yemeni history',
					'Most important things you learnt during this cash for work project in terms of importance of historical sites']
		child = False
		x, y = np.ogrid[100:500, :600]
		mask = ((x - 300) / 2) ** 2 + ((y - 300) / 3) ** 2 > 100 ** 2
		mask = 255 * mask.astype(int)

		title1.title('Wordclouds for open questions')

		feature = st.selectbox(
				'Select the question for which you would like to visualize wordclouds of answers', quest_list)

# _____________________________ months difficult ______________________________________ #

		if feature == 'Reason why the selected month is difficult':
			months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
					  'November', 'December']
			feats=[i for i in data if 'reason' in i]
			col1, col2, col3 = st.columns([2, 1, 2])
			df = data[feats].applymap(lambda x : '' if x == '0' else x).copy()

			corpus=''
			for n in range(12):
				corpus += ' '.join(df[feats[n]])
				corpus = re.sub('[^A-Za-z ]', ' ', corpus)
				corpus = re.sub('\s+', ' ', corpus)
				corpus = corpus.lower()
			sw = st.multiselect('Select words you would like to remove from the wordclouds \n\n',
								[i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])

			col1, col3 = st.columns([2, 2])

			for n in range(12):
				col_corpus = ' '.join(df[feats[n]].dropna())
				col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
				col_corpus = re.sub('\s+', ' ', col_corpus)
				col_corpus = col_corpus.lower()
				if col_corpus == ' ' or col_corpus == '':
					col_corpus = 'No_response'
				else:
					col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
				wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc.generate(col_corpus)
				if n%2 == 0:
					col1.subheader(months[n])
					col1.image(wc.to_array(), use_column_width=True)
				else:
					col3.subheader(months[n])
					col3.image(wc.to_array(), use_column_width=True)
# __________________________________ Learnings from Project _______________________________________________#
		elif feature in quest_list[-2:]:
			if 'Yemeni' in feature:
				colonnes=['learning1', 'learning2', 'learning3']
				titles=['First', 'Second', 'Third']
			else:
				colonnes = ['protection_learning1', 'protection_learning2', 'protection_learning3']
				titles = ['First', 'Second', 'Third']
			df = data.copy()
			corpus = ' '.join(data[colonnes[0]].dropna()) + \
					 ' '.join(data[colonnes[1]].dropna()) + ' '.join(data[colonnes[2]].dropna())
			corpus = re.sub('[^A-Za-z ]', ' ', corpus)
			corpus = re.sub('\s+', ' ', corpus)
			corpus = corpus.lower()
			sw = st.multiselect('Select words you would like to remove from the wordclouds \n\n',
								[i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
			col1, col2, col3 = st.columns([1, 1, 1])
			for i in range(3):
				col_corpus = ' '.join(df[colonnes[i]].dropna())
				col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
				col_corpus = re.sub('\s+', ' ', col_corpus)
				col_corpus = col_corpus.lower()
				if col_corpus == ' ' or col_corpus == '':
					col_corpus = 'No_response'
				else:
					col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
				wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc.generate(col_corpus)
				if i == 0:
					col1.subheader(titles[0])
					col1.image(wc.to_array(), use_column_width=True)
				elif i == 1:
					col2.subheader(titles[1])
					col2.image(wc.to_array(), use_column_width=True)
				else:
					col3.subheader(titles[2])
					col3.image(wc.to_array(), use_column_width=True)
			if st.checkbox('Would you like to filter Wordcloud according to other questions'):
				feature2 = st.selectbox('Select one question to filter the wordcloud',
										[questions[i] for i in data if i not in text and i != 'UniqueID' and i not in to_drop])
				filter2 = [i for i in questions if questions[i] == feature2][0]
				if filter2 in continues:
					a = data[filter2].astype(float).copy()
					minimum = st.slider('Select the minimum value you want to visulize',
										min_value=float(a.min()),
										max_value=float(a.max()),
										value=float(a.min())
										)
					maximum = st.slider('Select the maximum value you want to visulize', min_value=minimum,
										max_value=a.max(),value=a.max())
					df = data[(data[filter2] >= minimum) & (data[filter2] <= maximum)]
				else:
					filter3 = st.multiselect('Select the responses you want to include',
											 [i for i in data[filter2].unique()])
					df = data[data[filter2].isin(filter3)]
				#st.write(colonnes)
				col1, col2, col3 = st.columns([1, 1, 1])
				for i in range(3):
					col_corpus = ' '.join(df[colonnes[i]].dropna())
					col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
					col_corpus = re.sub('\s+', ' ', col_corpus)
					col_corpus = col_corpus.lower()
					if col_corpus == ' ' or col_corpus == '':
						col_corpus = 'No_response'
					else:
						col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
					wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
					wc.generate(col_corpus)
					if i == 0:
						col1.subheader('Main learning')
						col1.image(wc.to_array(), use_column_width=True)
					elif i == 1:
						col2.subheader('Second main learning')
						col2.image(wc.to_array(), use_column_width=True)
					else:
						col3.subheader('Third main learning')
						col3.image(wc.to_array(), use_column_width=True)
		else:
			d = {'Do you know why you were selected to participate in this project ?' : 'why',
				'Which type of training would you like to receive?' : 'trainings',
				'Explain which changes occurred in your life thanks to the CFW' : 'changes',
				'Explain what happened to others youth in similar needs of CFW who did not access the program' : 'youth',
				'Enter livelihood category' : 'Livelihood category',
				'Which skills did you learn ?' : 'skills',
				'What are you doing NOW in terms of incomes generation?' : 'income_generation',
				'What would stop you to do more of this work ?' : 'More_work_no_explain'}
			col_corpus = ' '.join(data[d[feature]].apply(lambda x : '' if x in ['I do not know', 'There is no', 'None']
																		else x).dropna())
			col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
			col_corpus = re.sub('\s+', ' ', col_corpus)
			col_corpus = col_corpus.lower()
			sw = st.multiselect('Select words you would like to remove from the wordclouds \n\n',
								[i[0] for i in Counter(col_corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
			if col_corpus == ' ' or col_corpus == '':
				col_corpus = 'No_response'
			else:
				col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
			wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
			wc.generate(col_corpus)
			col1, col2, col3 = st.columns([1, 4, 1])
			col2.image(wc.to_array(), use_column_width=True)

			if st.checkbox('Would you like to filter Wordcloud according to other questions'):
				feature2 = st.selectbox('Select one question to filter the wordcloud',
										[questions[i] for i in data if
										 i not in text and i != 'UniqueID' and i not in to_drop])
				filter2 = [i for i in questions if questions[i] == feature2][0]
				if filter2 in continues:
					a=data[filter2].astype(float)
					threshold = st.slider('Select threshold value you want to visualize',
										min_value=float(a.min()),
										max_value=float(a.max()),
										value=float(a.min())
										)
					DF=[data[data[filter2] <= threshold][d[feature]], data[data[filter2] > threshold][d[feature]]]
					titres=['Response under '+str(threshold),'Response over '+str(threshold)]
				else:
					DF=[data[data[filter2] == j][d[feature]] for j in data[filter2].unique()]
					titres=['Responded : '+j for j in data[filter2].unique()]
				col1, col2 = st.columns([1, 1])
				for i in range(len(DF)):
					col_corpus = ' '.join(DF[i].dropna())
					col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
					col_corpus = re.sub('\s+', ' ', col_corpus)
					col_corpus = col_corpus.lower()
					if col_corpus == ' ' or col_corpus == '':
						col_corpus = 'No_response'
					else:
						col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
					wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
					wc.generate(col_corpus)
					if i % 2 == 0:
						col1.subheader(titres[i])
						col1.image(wc.to_array(), use_column_width=True)
					else:
						col2.subheader(titres[i])
						col2.image(wc.to_array(), use_column_width=True)

if __name__ == '__main__':
	main()
