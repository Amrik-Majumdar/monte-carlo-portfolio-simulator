import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("MONTE CARLO PORTFOLIO SIMULATOR")
print("="*80 + "\n")

class DataLoader:
    
    @staticmethod
    def download_yfinance(tickers, start_date='2020-01-01', end_date=None):
        try:
            import yfinance as yf
            print(f"Fetching data for: {', '.join(tickers)}")
            print(f"Date range: {start_date} to {end_date or 'today'}")
            
            data = yf.download(tickers, start=start_date, end=end_date, 
                             progress=False, group_by='ticker')
            
            if data.empty:
                print("Error: No data received")
                return None
            
            prices = pd.DataFrame()
            
            if len(tickers) == 1:
                ticker = tickers[0]
                if 'Adj Close' in data.columns:
                    prices[ticker] = data['Adj Close']
                elif 'Close' in data.columns:
                    prices[ticker] = data['Close']
                else:
                    print(f"Error: No price data found")
                    return None
            else:
                for ticker in tickers:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            if 'Adj Close' in data[ticker].columns:
                                prices[ticker] = data[ticker]['Adj Close']
                            elif 'Close' in data[ticker].columns:
                                prices[ticker] = data[ticker]['Close']
                        else:
                            print(f"Warning: {ticker} not in dataset")
                    except Exception as e:
                        print(f"Warning: Issue with {ticker}: {e}")
            
            if prices.empty:
                print("Error: Failed to extract price data")
                return None
            
            prices = prices.dropna()
            
            if len(prices) < 100:
                print(f"Warning: Only {len(prices)} days available")
            
            print(f"Downloaded {len(prices)} days for {len(prices.columns)} assets")
            print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()}")
            
            return prices
            
        except ImportError:
            print("Error: yfinance not installed")
            print("Install: pip install yfinance")
            return None
        except Exception as e:
            print(f"Error downloading: {e}")
            return None
    
    @staticmethod
    def load_csv(filepath, date_column='Date'):
        try:
            print(f"Loading: {filepath}")
            df = pd.read_csv(filepath, parse_dates=[date_column], index_col=date_column)
            print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    @staticmethod
    def sample_portfolio():
        tickers = ['SPY', 'EFA', 'AGG', 'GLD', 'VNQ']
        return DataLoader.download_yfinance(tickers, start_date='2015-01-01')
    
    @staticmethod
    def generate_sample(n_days=1000):
        print("\nGenerating sample data...")
        
        np.random.seed(42)
        assets = ['US_Equities', 'Intl_Equities', 'Bonds', 'Commodities', 'REITs']
        
        mu = np.array([0.10, 0.08, 0.03, 0.05, 0.07])
        sigma = np.array([0.18, 0.20, 0.05, 0.25, 0.22])
        
        corr = np.array([
            [1.00, 0.75, 0.15, 0.30, 0.60],
            [0.75, 1.00, 0.10, 0.35, 0.55],
            [0.15, 0.10, 1.00, -0.05, 0.20],
            [0.30, 0.35, -0.05, 1.00, 0.40],
            [0.60, 0.55, 0.20, 0.40, 1.00]
        ])
        
        daily_mu = mu / 252
        daily_sigma = sigma / np.sqrt(252)
        
        L = np.linalg.cholesky(corr)
        
        prices = np.zeros((n_days, len(assets)))
        prices[0] = 100
        
        for t in range(1, n_days):
            Z = np.random.standard_normal(len(assets))
            Z_corr = L @ Z
            returns = daily_mu + daily_sigma * Z_corr
            prices[t] = prices[t-1] * (1 + returns)
        
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='B')
        df = pd.DataFrame(prices, columns=assets, index=dates)
        
        print(f"Generated {n_days} days of data")
        return df


class PortfolioEngine:
    
    def __init__(self, prices):
        if prices is None or prices.empty:
            raise ValueError("No price data provided")
        
        self.prices = prices.dropna()
        self.returns = self.prices.pct_change().dropna()
        self.assets = list(self.prices.columns)
        
        self.params = {}
        self.paths = None
        self.weights = None
        
        print(f"Initialized with {len(self.assets)} assets")
        print(f"Assets: {', '.join(self.assets)}")
        print(f"Period: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")
        print(f"Trading days: {len(self.returns)}")
    
    def analyze_history(self):
        print("\n" + "="*80)
        print("HISTORICAL PERFORMANCE")
        print("="*80)
        
        annual_ret = self.returns.mean() * 252
        annual_vol = self.returns.std() * np.sqrt(252)
        sharpe = (annual_ret - 0.02) / annual_vol
        
        corr = self.returns.corr()
        
        print("\nAsset Statistics (Annualized):")
        print("-" * 80)
        print(f"{'Asset':<15} {'Return':>10} {'Vol':>10} {'Sharpe':>10} {'MaxDD':>10}")
        print("-" * 80)
        
        for asset in self.assets:
            ret = annual_ret[asset] * 100
            vol = annual_vol[asset] * 100
            sr = sharpe[asset]
            
            cumret = (1 + self.returns[asset]).cumprod()
            runmax = cumret.cummax()
            dd = (cumret - runmax) / runmax
            maxdd = dd.min() * 100
            
            print(f"{asset:<15} {ret:>9.2f}% {vol:>9.2f}% {sr:>10.3f} {maxdd:>9.2f}%")
        
        print("\nCorrelation Matrix:")
        print("-" * 80)
        print(corr.round(2).to_string())
        
        return {'returns': annual_ret, 'vols': annual_vol, 'sharpe': sharpe, 'corr': corr}
    
    def fit_parameters(self, robust_cov=True):
        print("\n[Step 1/6] Fitting parameters...")
        
        mu = self.returns.mean() * 252
        
        if robust_cov:
            lw = LedoitWolf()
            cov = lw.fit(self.returns).covariance_ * 252
            print("Using Ledoit-Wolf covariance estimator")
        else:
            cov = self.returns.cov() * 252
        
        corr = self.returns.corr()
        sigma = np.sqrt(np.diag(cov))
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.returns)
        pca = PCA(n_components=min(5, len(self.assets)))
        pca.fit(scaled)
        
        regimes = self._detect_regimes(self.returns)
        
        self.params = {
            'mu': mu, 'cov': cov, 'corr': corr, 'sigma': sigma,
            'pca_comp': pca.components_, 'pca_var': pca.explained_variance_ratio_,
            'regimes': regimes
        }
        
        print(f"Fitted {len(mu)} assets")
        print(f"PCA variance explained: {pca.explained_variance_ratio_[:min(3, len(pca.explained_variance_ratio_))].round(3)}")
        
        return self
    
    def _detect_regimes(self, returns):
        vol = returns.std(axis=1).rolling(window=20).mean().dropna()
        threshold = vol.median()
        high_mask = vol > threshold
        
        regimes = {
            'high': {
                'mean': returns.loc[high_mask.index[high_mask]].mean(axis=1).mean(),
                'vol': returns.loc[high_mask.index[high_mask]].std(axis=1).mean(),
                'prob': high_mask.sum() / len(high_mask)
            },
            'low': {
                'mean': returns.loc[high_mask.index[~high_mask]].mean(axis=1).mean(),
                'vol': returns.loc[high_mask.index[~high_mask]].std(axis=1).mean(),
                'prob': (~high_mask).sum() / len(high_mask)
            }
        }
        
        print(f"Regimes detected - High vol: {regimes['high']['prob']*100:.1f}%, Low vol: {regimes['low']['prob']*100:.1f}%")
        
        return regimes
    
    def optimize(self, method='sharpe', rf=0.02):
        if not self.params:
            raise ValueError("Run fit_parameters() first")
        
        print(f"\n[Step 2/6] Optimizing portfolio ({method})...")
        
        n = len(self.params['mu'])
        mu = self.params['mu'].values
        cov = self.params['cov']
        
        bounds = tuple((0, 1) for _ in range(n))
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if method == 'equal_weight':
            self.weights = np.ones(n) / n
        elif method == 'min_variance':
            result = minimize(lambda w: w.T @ cov @ w, np.ones(n)/n,
                            method='SLSQP', bounds=bounds, constraints=constraints)
            self.weights = result.x
        elif method == 'sharpe':
            result = minimize(lambda w: -(np.sum(w * mu) - rf) / np.sqrt(w.T @ cov @ w),
                            np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=constraints)
            self.weights = result.x
        elif method == 'max_return':
            result = minimize(lambda w: -np.sum(w * mu), np.ones(n)/n,
                            method='SLSQP', bounds=bounds, constraints=constraints)
            self.weights = result.x
        elif method == 'risk_parity':
            def objective(w):
                pv = np.sqrt(w.T @ cov @ w)
                mc = cov @ w / pv
                rc = w * mc
                target = pv / n
                return np.sum((rc - target) ** 2)
            result = minimize(objective, np.ones(n)/n, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            self.weights = result.x
        
        alloc = dict(zip(self.assets, (self.weights * 100).round(1)))
        print(f"Weights: {alloc}")
        
        port_ret = np.sum(self.weights * mu) * 100
        port_vol = np.sqrt(self.weights.T @ cov @ self.weights) * 100
        port_sharpe = (port_ret/100 - rf) / (port_vol/100)
        
        print(f"Expected: {port_ret:.2f}% return, {port_vol:.2f}% vol, {port_sharpe:.3f} Sharpe")
        
        return self
    
    def simulate(self, capital=1000000, n_sims=10000, n_days=252, 
                 tx_cost=0.001, rebal_freq=21, use_regimes=True):
        
        print(f"\n[Step 3/6] Running {n_sims:,} simulations over {n_days} days...")
        
        mu = self.params['mu'].values
        cov = self.params['cov']
        
        port_mu = np.sum(self.weights * mu)
        port_vol = np.sqrt(self.weights.T @ cov @ self.weights)
        
        daily_mu = port_mu / 252
        daily_vol = port_vol / np.sqrt(252)
        
        paths = np.zeros((n_sims, n_days + 1))
        paths[:, 0] = capital
        
        regimes = self.params['regimes']
        
        print("Simulating paths...", end='', flush=True)
        for sim in range(n_sims):
            if sim % 2000 == 0 and sim > 0:
                print(f"\r{sim}/{n_sims} ({sim/n_sims*100:.0f}%)", end='', flush=True)
                
            value = capital
            last_rebal = 0
            
            for day in range(1, n_days + 1):
                if use_regimes:
                    vol_mult = 1.3 if np.random.rand() < regimes['high']['prob'] else 0.8
                    vol = daily_vol * vol_mult
                else:
                    vol = daily_vol
                
                z = np.random.standard_normal()
                ret = daily_mu + vol * z
                value *= (1 + ret)
                
                if rebal_freq and (day - last_rebal) >= rebal_freq:
                    value *= (1 - tx_cost)
                    last_rebal = day
                
                paths[sim, day] = value
        
        print(f"\rCompleted {n_sims:,} simulations")
        self.paths = paths
        
        metrics = self._calc_metrics(paths, capital, port_mu, port_vol)
        
        print(f"Final value range: ${paths[:, -1].min():,.0f} - ${paths[:, -1].max():,.0f}")
        
        return metrics
    
    def _calc_metrics(self, paths, capital, exp_ret, exp_vol):
        
        print("\n[Step 4/6] Computing risk metrics...")
        
        final = paths[:, -1]
        returns = (final - capital) / capital
        
        mean_final = np.mean(final)
        median_final = np.median(final)
        
        var_90 = capital - np.percentile(final, 10)
        var_95 = capital - np.percentile(final, 5)
        var_99 = capital - np.percentile(final, 1)
        
        cvar_95 = capital - np.mean(final[final <= np.percentile(final, 5)])
        
        rf = 0.02
        excess = np.mean(returns) - rf
        sharpe = (excess / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        down_ret = returns[returns < 0]
        down_std = np.std(down_ret) if len(down_ret) > 0 else 0
        sortino = (excess / down_std) * np.sqrt(252) if down_std > 0 else 0
        
        cummax = np.maximum.accumulate(paths, axis=1)
        dd = (paths - cummax) / cummax
        max_dd_avg = np.mean(np.min(dd, axis=1)) * 100
        max_dd_worst = np.min(dd) * 100
        
        prob_profit = (np.sum(final > capital) / len(final)) * 100
        prob_10 = (np.sum(returns > 0.10) / len(returns)) * 100
        prob_20 = (np.sum(returns > 0.20) / len(returns)) * 100
        prob_loss10 = (np.sum(returns < -0.10) / len(returns)) * 100
        
        metrics = {
            'capital': capital, 'exp_ret': exp_ret * 100, 'exp_vol': exp_vol * 100,
            'mean_final': mean_final, 'median_final': median_final, 
            'std_final': np.std(final), 'min_final': final.min(), 'max_final': final.max(),
            'var_90': var_90, 'var_95': var_95, 'var_99': var_99, 'cvar_95': cvar_95,
            'sharpe': sharpe, 'sortino': sortino,
            'max_dd_avg': max_dd_avg, 'max_dd_worst': max_dd_worst,
            'prob_profit': prob_profit, 'prob_10': prob_10, 
            'prob_20': prob_20, 'prob_loss10': prob_loss10,
            'skew': stats.skew(returns), 'kurt': stats.kurtosis(returns)
        }
        
        print("Risk metrics calculated")
        return metrics
    
    def visualize(self, metrics):
        
        print("\n[Step 5/6] Creating visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        paths = self.paths
        final = paths[:, -1]
        returns = (final - metrics['capital']) / metrics['capital'] * 100
        
        ax = fig.add_subplot(gs[0, :2])
        n_show = min(100, len(paths))
        for i in range(n_show):
            ax.plot(paths[i], alpha=0.05, color='steelblue', linewidth=0.5)
        
        pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        ax.plot(pct[2], 'darkred', linewidth=2.5, label='Median', zorder=10)
        ax.fill_between(range(len(pct[0])), pct[0], pct[4],
                        alpha=0.3, color='lightblue', label='5-95%')
        ax.fill_between(range(len(pct[0])), pct[1], pct[3],
                        alpha=0.4, color='skyblue', label='25-75%')
        
        ax.axhline(metrics['capital'], color='green', linestyle='--', 
                  linewidth=2, label=f"Initial: ${metrics['capital']:,.0f}", alpha=0.7)
        
        ax.set_title('Simulation Paths', fontsize=14, fontweight='bold')
        ax.set_xlabel('Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'${x/1e6:.2f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
        
        ax = fig.add_subplot(gs[0, 2])
        colors = sns.color_palette("Set3", len(self.assets))
        wedges, texts, autotexts = ax.pie(self.weights, labels=self.assets,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        for txt in autotexts:
            txt.set_color('black')
            txt.set_fontweight('bold')
            txt.set_fontsize(9)
        ax.set_title('Allocation', fontsize=12, fontweight='bold')
        
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(final, bins=60, density=True, alpha=0.7, 
               color='skyblue', edgecolor='black', linewidth=0.5)
        ax.axvline(metrics['capital'], color='green', linestyle='--', linewidth=2, label='Initial')
        ax.axvline(metrics['mean_final'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Final Value Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Final Value ($)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        
        ax = fig.add_subplot(gs[1, 1])
        ax.hist(returns, bins=60, density=True, alpha=0.7, 
               color='lightcoral', edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='green', linestyle='--', linewidth=2)
        ax.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Density')
        
        ax = fig.add_subplot(gs[1, 2])
        med_path = np.median(paths, axis=0)
        cummax = np.maximum.accumulate(med_path)
        dd = (med_path - cummax) / cummax * 100
        ax.fill_between(range(len(dd)), dd, 0, alpha=0.6, color='salmon')
        ax.plot(dd, 'darkred', linewidth=1.5)
        ax.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax.set_xlabel('Days')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        ax = fig.add_subplot(gs[2, 0])
        var_labels = ['VaR 90%', 'VaR 95%', 'VaR 99%']
        var_vals = [metrics['var_90'], metrics['var_95'], metrics['var_99']]
        bars = ax.barh(var_labels, var_vals, color=['yellow', 'orange', 'red'], 
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.set_title('Value at Risk', fontsize=12, fontweight='bold')
        ax.set_xlabel('VaR ($)')
        ax.grid(True, alpha=0.3, axis='x')
        for i, bar in enumerate(bars):
            w = bar.get_width()
            ax.text(w, bar.get_y() + bar.get_height()/2, 
                   f'${w:,.0f}', ha='left', va='center', fontweight='bold', fontsize=9)
        
        ax = fig.add_subplot(gs[2, 1])
        ratio_labels = ['Sharpe', 'Sortino']
        ratio_vals = [metrics['sharpe'], metrics['sortino']]
        colors = ['green' if v > 1 else 'orange' for v in ratio_vals]
        bars = ax.bar(ratio_labels, ratio_vals, color=colors, 
                     edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.axhline(1, color='blue', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_title('Performance Ratios', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ratio')
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                   f'{h:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax = fig.add_subplot(gs[2, 2])
        prob_labels = ['Profit', '10% Gain', '20% Gain', '10% Loss']
        prob_vals = [metrics['prob_profit'], metrics['prob_10'], 
                    metrics['prob_20'], metrics['prob_loss10']]
        colors = ['green', 'lightgreen', 'darkgreen', 'red']
        bars = ax.bar(prob_labels, prob_vals, color=colors, 
                     edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.set_title('Probabilities', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability (%)')
        ax.set_xticklabels(prob_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                   f'{h:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.suptitle('Monte Carlo Portfolio Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        try:
            plt.savefig('monte_carlo_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved: monte_carlo_analysis.png")
        except Exception as e:
            print(f"Could not save plot: {e}")
        
        plt.close()
    
    def report(self, metrics):
        
        print("\n[Step 6/6] Generating report...")
        print("\n" + "="*80)
        print("SIMULATION RESULTS")
        print("="*80)
        
        print("\nPortfolio Configuration")
        print("-" * 80)
        print(f"Assets: {', '.join(self.assets)}")
        print(f"Initial Capital: ${metrics['capital']:,.2f}")
        print(f"Expected Return: {metrics['exp_ret']:.2f}%")
        print(f"Expected Volatility: {metrics['exp_vol']:.2f}%")
        
        print("\nOutcomes")
        print("-" * 80)
        print(f"Mean Final Value: ${metrics['mean_final']:,.2f}")
        print(f"Median Final Value: ${metrics['median_final']:,.2f}")
        print(f"Std Dev: ${metrics['std_final']:,.2f}")
        print(f"Min: ${metrics['min_final']:,.2f}")
        print(f"Max: ${metrics['max_final']:,.2f}")
        
        profit = metrics['mean_final'] - metrics['capital']
        profit_pct = (profit / metrics['capital']) * 100
        print(f"\nExpected Profit: ${profit:,.2f} ({profit_pct:+.2f}%)")
        
        print("\nRisk Metrics")
        print("-" * 80)
        print(f"VaR 90%: ${metrics['var_90']:,.2f} ({metrics['var_90']/metrics['capital']*100:.2f}%)")
        print(f"VaR 95%: ${metrics['var_95']:,.2f} ({metrics['var_95']/metrics['capital']*100:.2f}%)")
        print(f"VaR 99%: ${metrics['var_99']:,.2f} ({metrics['var_99']/metrics['capital']*100:.2f}%)")
        print(f"CVaR 95%: ${metrics['cvar_95']:,.2f}")
        print(f"Max DD (avg): {metrics['max_dd_avg']:.2f}%")
        print(f"Max DD (worst): {metrics['max_dd_worst']:.2f}%")
        
        print("\nPerformance")
        print("-" * 80)
        print(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
        print(f"Sortino Ratio: {metrics['sortino']:.3f}")
        
        print("\nProbabilities")
        print("-" * 80)
        print(f"Prob of Profit: {metrics['prob_profit']:.2f}%")
        print(f"Prob of 10%+ Gain: {metrics['prob_10']:.2f}%")
        print(f"Prob of 20%+ Gain: {metrics['prob_20']:.2f}%")
        print(f"Prob of 10%+ Loss: {metrics['prob_loss10']:.2f}%")
        
        print("\nDistribution")
        print("-" * 80)
        skew_txt = 'right-skewed' if metrics['skew'] > 0 else 'left-skewed'
        kurt_txt = 'fat-tailed' if metrics['kurt'] > 0 else 'thin-tailed'
        print(f"Skewness: {metrics['skew']:.3f} ({skew_txt})")
        print(f"Kurtosis: {metrics['kurt']:.3f} ({kurt_txt})")
        
        print("\n" + "="*80)
        print("DONE")
        print("="*80)


def main():
    
    print("Data Source:")
    print("1. Download ETFs (SPY, EFA, AGG, GLD, VNQ)")
    print("2. Custom tickers")
    print("3. Load CSV")
    print("4. Generated sample")
    
    choice = input("\nSelect (1-4): ").strip()
    
    data = None
    
    if choice == '1':
        print("\nFetching ETF portfolio...")
        data = DataLoader.sample_portfolio()
        if data is None or data.empty:
            print("Falling back to sample data")
            data = DataLoader.generate_sample()
    
    elif choice == '2':
        tickers = input("\nTickers (comma-separated): ").strip()
        tickers = [t.strip().upper() for t in tickers.split(',')]
        start = input("Start date (YYYY-MM-DD, default 2015-01-01): ").strip() or '2015-01-01'
        data = DataLoader.download_yfinance(tickers, start_date=start)
        if data is None or data.empty:
            print("Falling back to sample data")
            data = DataLoader.generate_sample()
    
    elif choice == '3':
        path = input("\nCSV path: ").strip()
        data = DataLoader.load_csv(path)
    
    elif choice == '4':
        print("\nGenerating sample...")
        data = DataLoader.generate_sample()
    
    else:
        print("Invalid choice, using sample")
        data = DataLoader.generate_sample()
    
    if data is None or data.empty:
        print("\nFailed to load data")
        return
    
    print("\n" + "="*80)
    
    engine = PortfolioEngine(data)
    engine.analyze_history()
    engine.fit_parameters(robust_cov=True)
    
    print("\nOptimization Method:")
    print("1. Max Sharpe (recommended)")
    print("2. Min Variance")
    print("3. Risk Parity")
    print("4. Equal Weight")
    print("5. Max Return")
    
    opt = input("\nSelect (1-5, default 1): ").strip() or '1'
    
    methods = {'1': 'sharpe', '2': 'min_variance', '3': 'risk_parity', 
               '4': 'equal_weight', '5': 'max_return'}
    method = methods.get(opt, 'sharpe')
    
    engine.optimize(method=method)
    
    print("\nSimulation Parameters:")
    
    try:
        cap = float(input("Capital (default 1000000): ").strip() or 1000000)
        sims = int(input("Simulations (default 10000): ").strip() or 10000)
        days = int(input("Days (default 252): ").strip() or 252)
    except ValueError:
        print("Invalid input, using defaults")
        cap = 1000000
        sims = 10000
        days = 252
    
    metrics = engine.simulate(capital=cap, n_sims=sims, n_days=days,
                              tx_cost=0.001, rebal_freq=21, use_regimes=True)
    
    engine.visualize(metrics)
    engine.report(metrics)
    
    print("\n" + "="*80)
    print("SAVING FILES")
    print("="*80)
    
    try:
        df_metrics = pd.DataFrame([metrics]).T
        df_metrics.columns = ['Value']
        df_metrics.to_csv('metrics.csv')
        print("Saved: metrics.csv")
        
        df_weights = pd.DataFrame({
            'Asset': engine.assets,
            'Weight': engine.weights,
            'Percent': engine.weights * 100
        })
        df_weights.to_csv('weights.csv', index=False)
        print("Saved: weights.csv")
        
        n_sample = min(100, len(engine.paths))
        df_paths = pd.DataFrame(engine.paths[:n_sample].T)
        df_paths.columns = [f'Path_{i+1}' for i in range(n_sample)]
        df_paths.to_csv('paths.csv', index=False)
        print("Saved: paths.csv")
        
    except Exception as e:
        print(f"Error saving: {e}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print("  monte_carlo_analysis.png")
    print("  metrics.csv")
    print("  weights.csv")
    print("  paths.csv")
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")