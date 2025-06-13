
from src.data_loader import load_data
from sklearn.linear_model import LinearRegression

def main():
    df = load_data()
    print("\nShape of Data:", df.shape)
    print("\n Column names:\n", df.columns.tolist())
    print("\n Data Types: \n", df.dtypes)
    print("\n Null Values: \n", df.isnull().sum())
    print("\n Sample Data:")
    print("\n First 5 Rows: \n",df.head())
    print("\n Last 5 Rows:\n", df.tail())
    print("\n Summary Statistics: \n", df.describe())

    #Basic correlation check
    print("\n Correlation Matrix: \n", df.corr(numeric_only=True))



    #Plot price distribution 
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8,5))
    sns.histplot(df['Price'], kde=True, color='skyblue')
    plt.title("Distribution of Housing Price")
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Plot Features vs Price
    
    features = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Area Population']

    for col in features:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df["Price"], alpha=0.5)
        plt.title(f'{col} vs Price')
        plt.xlabel(col)
        plt.ylabel("Price")
        plt.tight_layout()
        plt.show()


    # Correlation Heatmap

    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


    # Data Cleaning

    df = df.drop('Address', axis=1)

    #- Null Values Check

    print("\n, Null values in Dataset:\n")
    print(df.isnull().sum())      #if 0 comes on all places means perfect, otherwise we use dropna(), fillna().

    # Feature/Target Split

    x = df.drop('Price', axis=1)
    y = df['Price']


    #Train/Test Split

    from sklearn.model_selection import train_test_split 

    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    # x - independent variable(feature/input)- like area income, rooms,etc
    # y - dependent variable(target/output) - here it's 'Price'
    
    
    print("Training set shape:", x_train.shape)
    print("Test set shape:", x_test.shape)

    #    # Model Initialization & Training
    model = LinearRegression()   #model initialization
    model.fit(x_train, y_train)  # Trainig the model
    print("\n Model training complete!")

    
    # # Predict on Test Set
    y_pred = model.predict(x_test)
    print("\n Sample Predictions: ")
    print(y_pred[:5])


    from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score


    #Evaluation matrices
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5 
    r2 = r2_score(y_test, y_pred)  

    print("\n Model Evaluation Matrices:")
    print(f'MAE(mean absolute error):{mae:.2f}')
    print(f'MSE(mean squared error):{mse:.2f}')
    print(f'RMSE(root mean squared error):{rmse:.2f}')
    print(f'R2 score:{r2:.2f}')


    #Plot Actual vs Predicted Prices
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha= 0.6, color='green')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Housing Prices")
    #reference line 
    plt.plot([y_test.min(),y_test.max()], [y_test.min(), y_test.max()] , '--r')  
    plt.tight_layout()
    plt.show()


        # Save Trained Model
    import joblib
    joblib.dump(model, "house_price_model.pkl")
    print("\n Model saved successfully as 'house_price_model.pkl'")




if __name__ == "__main__":
    main()


