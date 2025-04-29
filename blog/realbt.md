Note: REALBT is my 101st project, and something I wanted to do from a long time. It is not very polished, and there are scopes of improvement. Contributions and suggestions are always welcome.

# REALBT - A Simple Back-testing Engine in Pure Python

Repo: [yash-srivastava19/realbt](https://github.com/yash-srivastava19/REALBT)

## Finance huh, are you THAT guy?

I never thought I will work in finance, because I've always learned AI. Finance as a field has always fascinated me, because it was really mathematics heavy, and I really liked the stochastic + logical nature of it. Since my last two two roles were in fintech, I thought it is high time I learn more about finance, and what better way to learn than make something that can be used by the community, and hence REALBT was developed to deeper my understanding of finance fundamentals. REALBT is a back-testing engine written purely in python, and offers several advantages over existing back-testing engines.

## What exactly is Back-Testing?

*"The idea that the future is predictable is a very foolish idea."*

As mentioned, finance is both stochastic and logical, and if you want to make a fortune, you need to play your cards well. Now, life would've been easier(but chaotic) if there were a mathematical model that just churns out correct stock prices everytime, but reality is far from it. Markets are very unpredictable, and placing your bets on stocks is pretty much comparable to gambling, but that is not always true if you have the right tools in your arsenal - and that's exactly what a back-testing engine does. An analogy to better understand what exactly back-testing engine does is this: 

*Imagine before buying a real car(a big investment), you buy a toy car to know whether your investment is worth it or not. A back-test is basically testing your toy car on a pretend road to see how well it drives. Now, a good condition test for the toy car would be when the pretend road is like a real road - with bumps, traffic, gas stations etc. In this way, when your toy car does well on the pretend road, you can be sure that the real car will also do well.*

That's exactly a backtesting engine like REALBT does. You can play with your trading strategy on past market data to see how well it performs. This gives you confidence about your trading strategy in the real world, so instead of playing a gamble, you make an educated choice about which stock to invest in or not. REALBT is made in an extensible fashion, so more robust strategies and more stocks coverage can be done in the future for the community to use.

## What are REALBT's unique features?
REALBT (REAListic BackTesting) is made exactly for the reasons highlighted above. It is a Python-based framework designed for realistic backtesting of trading strategies. Unlike traditional backtesting frameworks, it emphasizes accurate modeling of market frictions such as **slippage**, **market impact**, and **transaction costs**. The package is made keeping in mind the requirements of traders and researchers who want to evaluate strategies under conditions that closely mimic real-world trading environments.

REALBT is modular, extensible, written fully in python and user-friendly, with a command-line interface (CLI) for ease of use. It also includes tools for data fetching, strategy creation, and result visualization - similar to existing solutions for backtesting. Here are some of the unique features of REALBT:

- **Realistic Market Friction Modeling** REALBT incorporates modules to simulate market frictions:
    
    - **Slippage**: Defined in `costs/liquidity.py`
    - **Market Impact**: Defined in `costs/market_impact.py`
    - **Transaction Costs**: Defined in `costs/transaction_cost.py`
    
    In other Backtest engines such as `backtesting.py`, the ability to add commission(as transaction cost) is there, but there is no option for market impact and slippage, making REALBT more rich and closely mimic real world scenarios. These modules allow users to account for the hidden costs of trading, providing a more accurate evaluation of strategy performance.
   
```python
# Example: Calculating transaction costs    
from costs.transaction_cost import calculate_transaction_cost
trade_volume = 1000
price_per_unit = 150
transaction_cost = calculate_transaction_cost(trade_volume, price_per_unit) 
print(f"Transaction Cost: ${transaction_cost}")
```

- **Command-Line Interface (CLI)** The CLI, implemented in `cli.py`, Key commands include:
    - `new`: Creates a new project with sample files.
    - `fetch-data`: Fetches historical stock data.
    - `run`: Executes a backtest based on a configuration file.
    
	REALBT comes in pre-packaged with CLI, that makes it easier to run backtest from CLI, something that is not available in major backtest engines. The commands are pretty straightforward.
	
    Example of creating a new project:
	```bash
python realbt/cli.py new my_project -d /path/to/directory
	```
	
    Internally, the `create_new_project` function generates the required folder structure and sample files:

```python
@cli.command("new")
@click.argument("project_name")
@click.option("--directory", "-d", default=".", help="Target directory")
def create_new_project(project_name, directory):
    project_dir = os.path.join(directory, project_name)
    os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "strategies"), exist_ok=True)
    with open(os.path.join(project_dir, "strategies", "sample_strategy.py"), "w") as f:
        f.write("from realbt.src.engine import BacktestEngine\n\n# Define your strategy here")

```

- **Data Fetching** The `fetch-data` command, implemented in `cli.py`, uses the `fetch_stock_data` function to retrieve historical stock data. This data is stored in CSV format for easy access. The data is provided by `yfinance` , and uses few parameters from the OHLCV to run the backtest engine.

```python
@cli.command("fetch-data")
@click.argument("ticker")
@click.argument("start_date")
@click.argument("end_date")
@click.argument("output_file", type=click.Path())
def fetch_data(ticker, start_date, end_date, output_file):
    data = fetch_stock_data(ticker, start_date, end_date)
    data.to_csv(output_file, index=False)
```
    
- **Visualization** REALBT integrates with Plotly for interactive visualizations. For example, portfolio performance over time can be visualized using the `_generate_report` function in `cli.py` . The inspiration for the plot is taken from the `backtesting.py` ,but the implementation is in Plotly(`backtesting.py` uses bokeh for plotting). This is an example `backtesting.py` plot, and REALBT has similar plot to this:

![bokeh_plot](https://github.com/user-attachments/assets/264b140d-68da-49c2-a8f0-d9396d932929)

```python
fig = px.line(data_frame=portfolio_data, x="time", y="value", title="Portfolio Value")
fig.show()
```
    
- **Extensibility** The modular design allows users to extend the framework by adding custom cost models, strategies, or data sources. For instance, a new cost model can be added by creating a Python file in the `costs` directory and integrating it into the backtesting engine. The core idea behind REALBT is for researchers and traders to extend this framework and make it more useful for the community. Contributions and suggestions are always welcome. Raise a PR and get in [touch](mailto:ysrivastava82@gmail.com)

## How to use REALBT?

- **Clone the Repository:** First, clone the repository to your local machine using Git.
```bash
git clone https://github.com/username/realbt.git && cd realbt
```

- **Set Up the Environment**

REALBT has dependencies that need to be installed. It is recommended to use a virtual environment to avoid conflicts with other Python packages.

- **Create a Virtual Environment**:
```python
python -m venv venv
```

- **Activate the Virtual Environment**:
- On Windows:
```bash
venv\Scripts\activate
```

- On macos/Linux
```bash
source venv/bin/activate
```

- **Install Dependencies**
```bash
pip install -r requirements.txt
```
    
- **Verify Installation**
Run the CLI help command to ensure the package is installed and working correctly:

```bash
python realbt/cli.py --help
```

You should see a list of available commands, such as `new`, `fetch-data`, and `run`.

- **Create a New Project:** Use the `new` command to create a new backtesting project:

```bash
python realbt/cli.py new my_project -d /path/to/directory
```


This will generate the following folder structure:
```
my_project/
├── data/
├── results/
├── strategies/
│   └── sample_strategy.py
└── config.yaml
```


 5. **Fetch Historical Data**

Fetch historical stock data using the `fetch-data` command. For example, to fetch Apple stock data for 2024:

```bash
python realbt/cli.py fetch-data AAPL 2024-01-01 2024-12-31 my_project/data/apple.csv
```


---

- **Define a Strategy**

Edit the `sample_strategy.py` file in the `strategies` folder to define your trading strategy. For example:

```python

from realbt.src.engine import BacktestEngine
def my_strategy(data):
    # Example: Moving Average Crossover Strategy
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    signals = data['SMA_50'] > data['SMA_200']
    return signals

engine = BacktestEngine()
engine.run(data_path="data/apple.csv", strategy=my_strategy)
```


---

- **Run the Backtest**

Run the Backtest using the `run` command:

```bash
python realbt/cli.py run my_project/config.yaml
```


The results will be saved in the `results` folder, and you can visualize them using the built-in visualization tools.

---

- **Extend the Framework**

REALBT is modular and extensible. You can:

- Add custom cost models in the `costs` directory.
- Create new strategies in the `strategies` folder.
- Modify the backtesting engine to suit specific requirements.

For example, to add a custom transaction cost model:

```python
def custom_transaction_cost(volume, price):
    return 0.001 * volume * price  # Example: 0.1% transaction cost
```

Integrate it into your strategy:

```python
from realbt.costs.custom_transaction_cost import custom_transaction_cost
def my_strategy_with_costs(data):
    # Define strategy logic
    ...
    # Apply custom transaction costs
    costs = custom_transaction_cost(volume, price)
    ...
```

**With your support and contributions, REALBT can be made better! Make a PR today and get in [touch](mailto:ysrivastava82@gmail.com).**
