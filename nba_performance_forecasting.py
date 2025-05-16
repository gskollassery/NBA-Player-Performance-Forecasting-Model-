import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/nba_stats.csv"
MODEL_PATH = "models/performance_model.pkl"
SALARY_DATA_PATH = "data/nba_salaries.csv"

class NBAPerformanceForecaster:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importances = None
        self.undervalued_players = None
        
    def load_data(self):
        """Load and preprocess NBA data"""
        try:
            
            stats_df = pd.read_csv(DATA_PATH)
            
        
            try:
                salary_df = pd.read_csv(SALARY_DATA_PATH)
                stats_df = pd.merge(stats_df, salary_df, on='player_id', how='left')
            except FileNotFoundError:
                print("Salary data not found - proceeding without salary analysis")
                stats_df['salary'] = np.nan
            
            stats_df['PER'] = self.calculate_per(stats_df)
            stats_df['TS%'] = self.calculate_true_shooting(stats_df)
            stats_df['USG%'] = self.calculate_usage(stats_df)
    
            stats_df.fillna(stats_df.median(), inplace=True)
            
            return stats_df
        
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    @staticmethod
    def calculate_per(df):
        """Calculate Player Efficiency Rating"""
        return ((df['PTS'] + df['TRB'] + df['AST'] + df['STL'] + df['BLK'] - 
                (df['FGA'] - df['FG']) - (df['FTA'] - df['FT']) - df['TOV']) / df['MP']
    
    @staticmethod
    def calculate_true_shooting(df):
        """Calculate True Shooting Percentage"""
        return df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
    
    @staticmethod
    def calculate_usage(df):
        """Calculate Usage Percentage"""
        return (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / (
            df['MP'] * (df['team_FGA'] + 0.44 * df['team_FTA'] + df['team_TOV']))
    
    def train_model(self, df, target='PER'):
        """Train performance prediction model"""
        try:
            features = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 
                       'MP', 'TS%', 'USG%', 'age', 'experience']
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.model = make_pipeline(
                StandardScaler(),
                RandomForestRegressor(n_estimators=100, random_state=42)
            )
            self.model.fit(X_train, y_train)
        
            y_pred = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"Model trained successfully:")
            print(f"- RMSE: {rmse:.3f}")
            print(f"- R2 Score: {r2:.3f}")
        
            self.feature_importances = pd.DataFrame({
                'Feature': features,
                'Importance': self.model.named_steps['randomforestregressor'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def identify_undervalued_players(self, df):
        """Identify undervalued players based on performance vs salary"""
        try:
            if 'salary' not in df.columns:
                print("Salary data not available for undervalued analysis")
                return False
                
            features = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 
                       'MP', 'TS%', 'USG%', 'age', 'experience']
            df['predicted_PER'] = self.model.predict(df[features])
            
            df['value_metric'] = df['predicted_PER'] / (df['salary'] / 1e6)
            
            self.undervalued_players = df.sort_values('value_metric', ascending=False).head(10)
            
            print("\nTop 5 Undervalued Players:")
            print(self.undervalued_players[['player_name', 'predicted_PER', 'salary', 'value_metric']].head(5))
            
            return True
            
        except Exception as e:
            print(f"Error in undervalued analysis: {str(e)}")
            return False
    
    def save_model(self):
        """Save trained model"""
        try:
            Path("models").mkdir(exist_ok=True)
            joblib.dump(self.model, MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def visualize_feature_importance(self):
        """Visualize feature importance"""
        if self.feature_importances is None:
            print("Train model first to get feature importances")
            return
            
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=self.feature_importances,
            palette='viridis'
        )
        plt.title("Feature Importance for Performance Prediction")
        plt.tight_layout()
        plt.savefig("visualization/feature_importance.png")
        plt.show()
    
    def generate_powerbi_data(self):
        """Prepare data for Power BI dashboard"""
        try:
            if self.undervalued_players is None:
                print("Run undervalued analysis first")
                return False
                
            output_cols = [
                'player_name', 'age', 'team', 'position', 'PTS', 'TRB', 'AST',
                'predicted_PER', 'salary', 'value_metric'
            ]
            powerbi_data = self.undervalued_players[output_cols]
            powerbi_data.to_csv("visualization/powerbi_data.csv", index=False)
            print("Power BI data exported to visualization/powerbi_data.csv")
            return True
            
        except Exception as e:
            print(f"Error generating Power BI data: {str(e)}")
            return False

def main():
    """Main execution function"""
    forecaster = NBAPerformanceForecaster()
    
    print("Loading data...")
    nba_data = forecaster.load_data()
    if nba_data is None:
        return
    
    print("\nTraining model...")
    if not forecaster.train_model(nba_data):
        return
    
    print("\nAnalyzing player value...")
    forecaster.identify_undervalued_players(nba_data)
    
    print("\nGenerating visualizations...")
    forecaster.visualize_feature_importance()
    
    print("\nPreparing Power BI data...")
    forecaster.generate_powerbi_data()
    
    print("\nSaving model...")
    forecaster.save_model()

if __name__ == "__main__":
    from pathlib import Path
    main()