import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import io
import base64  


# Load the dataset
df=pd.read_csv("./heart.csv")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('heart disease', axis=1), df['heart disease'], test_size=0.3, random_state=42)

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Decision Tree Hyperparameter Tuning"),
    
    html.Label("Criterion:"),
    dcc.Dropdown(
        id='criterion-dropdown',
        options=[
            {'label': 'Gini', 'value': 'gini'},
            {'label': 'Entropy', 'value': 'entropy'},
        ],
        value='gini',
        clearable=False
    ),
    
    html.Label("Max Depth:"),
    dcc.Dropdown(
        id='max-depth-dropdown',
        options=[{'label': str(i), 'value': i} for i in [2,5,10, 20, 30, 40]],
        value=None,
        clearable=False
    ),
    
    html.Label("Min Samples Split:"),
    dcc.Dropdown(
        id='min-samples-split-dropdown',
        options=[{'label': str(i), 'value': i} for i in [2, 5, 10]],
        value=2,
        clearable=False
    ),
    
    html.Label("Min Samples Leaf:"),
    dcc.Dropdown(
        id='min-samples-leaf-dropdown',
        options=[{'label': str(i), 'value': i} for i in [1, 2, 4]],
        value=1,
        clearable=False
    ),
    
    html.Button('Train Model', id='train-button', n_clicks=0),
    
    html.H3(id='accuracy-output', style={'margin-top': '20px'}),
    html.Img(id='tree-graph', style={'margin-top': '20px'}),
])

@app.callback(
    [Output('accuracy-output', 'children'),
     Output('tree-graph', 'src')],
    [Input('train-button', 'n_clicks')],
    [Input('criterion-dropdown', 'value'),
     Input('max-depth-dropdown', 'value'),
     Input('min-samples-split-dropdown', 'value'),
     Input('min-samples-leaf-dropdown', 'value')]
)
def update_output(n_clicks, criterion, max_depth, min_samples_split, min_samples_leaf):
    if n_clicks > 0:
        # Train the model with selected hyperparameters
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Plot the tree
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, filled=True, feature_names=df.columns[:-1], class_names=['No Disease', 'Disease'], ax=ax)
        
        # Save plot as PNG image and encode in base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        image_src = f'data:image/png;base64,{image_base64}'
        
        return f'Accuracy: {accuracy:.2f}', image_src
    return '', ''

if __name__ == '__main__':
    app.run_server(debug=True)