import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing, basinhopping
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class PriceOptimizer:
    """
    A comprehensive price optimization package for analytical_ride_type pricing decisions.
    
    This class handles data preparation, metric calculations, and optimization
    to maximize/minimize target metrics (delta_bookings, delta_pmm, delta_rides)
    subject to constraints.
    """
    
    def __init__(self, levels: Optional[List[str]] = None):
        """
        Initialize the PriceOptimizer.
        
        Args:
            levels: List of additional grouping levels beyond region-analytical_ride_type.
                   Can include 'use_case', 'db', or both.
                   Examples: ['use_case'], ['db'], ['use_case', 'db']
                   Default: None (aggregate to region-analytical_ride_type level only)
        """
        self.levels = levels or []
        self.history_df = None
        self.price_df = None
        self.merged_df = None
        self.results_df = None
        self.optimization_results = {}
        
        # Validate levels
        valid_levels = ['use_case', 'db']
        invalid_levels = set(self.levels) - set(valid_levels)
        if invalid_levels:
            raise ValueError(f"Invalid levels: {invalid_levels}. Valid levels are: {valid_levels}")
        
    def load_data(self, history_df: pd.DataFrame, price_df: pd.DataFrame) -> None:
        """
        Load and validate input dataframes.
        
        Args:
            history_df: Historical data with pmm, bookings, rides, elasticity
                       Can include optional 'use_case' and 'db' columns
            price_df: Price data with sessions, total_price
                     Can include optional 'use_case' and 'db' columns
                     
        Note:
            Data will be aggregated based on the levels specified during initialization.
            Base grouping is always region-analytical_ride_type + any additional levels.
            Elasticity will be rides-weighted during aggregation.
        """
        # Validate required columns
        required_history_cols = ['region', 'analytical_ride_type', 'pmm', 'bookings', 'rides', 'elasticity']
        required_price_cols = ['region', 'analytical_ride_type', 'sessions', 'total_price']
        
        missing_history = set(required_history_cols) - set(history_df.columns)
        missing_price = set(required_price_cols) - set(price_df.columns)
        
        if missing_history:
            raise ValueError(f"Missing columns in history_df: {missing_history}")
        if missing_price:
            raise ValueError(f"Missing columns in price_df: {missing_price}")
            
        self.history_df = history_df.copy()
        self.price_df = price_df.copy()
        
        # Merge dataframes
        self._merge_dataframes()
        
    def _merge_dataframes(self) -> None:
        """Merge history and price dataframes and calculate derived metrics."""
        # Define grouping columns based on levels parameter
        grouping_cols = ['region', 'analytical_ride_type'] + self.levels
        
        # Aggregate history data with specified levels
        history_agg = self._aggregate_history_data(self.history_df, grouping_cols)
        
        # Aggregate price data with specified levels
        price_agg = self._aggregate_price_data(self.price_df, grouping_cols)
        
        # Merge aggregated dataframes
        self.merged_df = pd.merge(
            history_agg, 
            price_agg, 
            on=grouping_cols, 
            how='inner'
        )
        
        # Calculate initial derived metrics
        self._calculate_base_metrics()
    
    def _aggregate_history_data(self, df: pd.DataFrame, grouping_cols: List[str]) -> pd.DataFrame:
        """
        Aggregate history data with rides-weighted elasticity.
        
        Args:
            df: History dataframe with detailed data
            grouping_cols: List of columns to group by
            
        Returns:
            Aggregated dataframe at specified granularity level
        """
        # Validate that required columns exist
        missing_cols = set(grouping_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in history_df for specified levels: {missing_cols}")
        
        # Calculate rides-weighted elasticity before aggregation
        df = df.copy()
        df['elasticity_rides_weighted'] = df['elasticity'] * df['rides']
        
        # Define aggregation dictionary
        agg_dict = {
            'pmm': 'sum',
            'bookings': 'sum', 
            'rides': 'sum',
            'elasticity_rides_weighted': 'sum'  # Sum of weighted values
        }
        
        # Add any remaining columns (not in grouping) as 'first'
        remaining_cols = set(df.columns) - set(grouping_cols) - set(agg_dict.keys()) - {'elasticity_rides_weighted'}
        for col in remaining_cols:
            if col not in ['elasticity']:  # Skip elasticity as we calculate it separately
                agg_dict[col] = 'first'
        
        # Perform aggregation
        history_agg = df.groupby(grouping_cols).agg(agg_dict).reset_index()
        
        # Calculate final rides-weighted elasticity
        history_agg['elasticity'] = (history_agg['elasticity_rides_weighted'] / 
                                   history_agg['rides'])
        
        # Drop the intermediate column
        history_agg = history_agg.drop('elasticity_rides_weighted', axis=1)
        
        return history_agg
    
    def _aggregate_price_data(self, df: pd.DataFrame, grouping_cols: List[str]) -> pd.DataFrame:
        """
        Aggregate price data at specified granularity.
        
        Args:
            df: Price dataframe with detailed data
            grouping_cols: List of columns to group by
            
        Returns:
            Aggregated dataframe at specified granularity level
        """
        # Validate that required columns exist
        missing_cols = set(grouping_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in price_df for specified levels: {missing_cols}")
        
        # Define aggregation dictionary
        agg_dict = {
            'sessions': 'sum',
            'total_price': 'sum'
        }
        
        # Add any remaining columns (not in grouping) as 'first'
        remaining_cols = set(df.columns) - set(grouping_cols) - set(agg_dict.keys())
        for col in remaining_cols:
            agg_dict[col] = 'first'
        
        # Perform aggregation
        price_agg = df.groupby(grouping_cols).agg(agg_dict).reset_index()
        
        return price_agg
        
    def _calculate_base_metrics(self) -> None:
        """Calculate base metrics for the merged dataframe."""
        df = self.merged_df
        
        # Base calculations
        df['avg_price'] = df['total_price'] / df['sessions']
        df['avg_pmm'] = df['pmm'] / df['rides']
        df['weekly_rides'] = df['rides'] / 4
        df['13w_rides'] = df['weekly_rides'] * 13
        df['13w_bookings'] = df['13w_rides'] * df['avg_price']
        df['13w_pmm'] = df['13w_rides'] * df['avg_pmm']
        
        # Initialize price change and derived metrics (will be updated during optimization)
        df['price_change'] = 0.0  # in basis points
        self._update_derived_metrics()
        
    def _update_derived_metrics(self) -> None:
        """Update derived metrics based on current price_change values."""
        df = self.merged_df
        
        # Calculate new metrics based on price changes
        df['new_price'] = df['avg_price'] * (1 + df['price_change'] / 10000)  # basis points to percentage
        
        # Calculate new rides based on elasticity
        price_change_pct = df['price_change'] / 10000
        rides_change_pct = df['elasticity'] * price_change_pct
        df['new_weekly_rides'] = df['weekly_rides'] * (1 + rides_change_pct)
        df['new_13w_rides'] = df['new_weekly_rides'] * 13
        
        # Calculate new bookings and pmms
        df['new_13w_bookings'] = df['new_price'] * df['new_13w_rides']
        df['new_avg_pmm'] = df['new_price'] - df['avg_price'] + df['avg_pmm']
        df['new_13w_pmm'] = df['new_avg_pmm'] * df['new_13w_rides']
        
        # Calculate deltas
        df['delta_bookings'] = df['new_13w_bookings'] - df['13w_bookings']
        df['delta_pmm'] = df['new_13w_pmm'] - df['13w_pmm']
        df['delta_rides'] = df['new_13w_rides'] - df['13w_rides']
        
        # Calculate price sensitivity metrics
        df['pset'] = np.where(df['delta_rides'] != 0, 
                             df['delta_bookings'] / df['delta_rides'], 0)
        df['pmet'] = np.where(df['delta_rides'] != 0, 
                             df['delta_pmm'] / df['delta_rides'], 0)
    
    def set_price_changes(self, price_changes: Union[np.ndarray, List[float]]) -> None:
        """
        Set price changes and update all derived metrics.
        
        Args:
            price_changes: Array of price changes in basis points
        """
        if len(price_changes) != len(self.merged_df):
            raise ValueError(f"Price changes length ({len(price_changes)}) must match dataframe length ({len(self.merged_df)})")
        
        self.merged_df['price_change'] = price_changes
        self._update_derived_metrics()
    
    def calculate_total_metric(self, target: str) -> float:
        """
        Calculate total value for a target metric.
        
        Args:
            target: One of 'delta_bookings', 'delta_pmm', 'delta_rides'
        
        Returns:
            Total value of the target metric
        """
        if target not in ['delta_bookings', 'delta_pmm', 'delta_rides']:
            raise ValueError("Target must be one of: delta_bookings, delta_pmm, delta_rides")
        
        return self.merged_df[target].sum()

    def _objective(self, price_changes, target, maximize):
        """Objective function to optimize, now a method of the class."""
        try:
            self.set_price_changes(price_changes)
            value = self.calculate_total_metric(target)
            return -value if maximize else value
        except Exception as e:
            # print(f"Error in objective function: {e}") # Keep silent to avoid spamming during optimization
            return 1e10  # Return large penalty for invalid solutions

    def _constraint_function(self, price_changes, constraints):
        """Evaluate all constraints, now a method of the class."""
        try:
            self.set_price_changes(price_changes)
            violations = []
            
            if constraints:
                for constraint in constraints:
                    metric_value = self.calculate_total_metric(constraint['metric'])
                    target_value = constraint['value']
                    operator = constraint['operator']
                    
                    if operator == '>=':
                        violations.append(metric_value - target_value)
                    elif operator == '<=':
                        violations.append(target_value - metric_value)
                    elif operator == '==':
                        violations.append(abs(metric_value - target_value))
            
            return violations
        except Exception as e:
            # print(f"Error in constraint function: {e}") # Keep silent
            return [-1e10] * len(constraints) if constraints else []

    def optimize_pricing(self, 
                        target: str, 
                        maximize: bool = True,
                        constraints: Optional[List[Dict]] = None,
                        price_bounds: Optional[Tuple[float, float]] = None,
                        price_direction: str = 'both',
                        method: str = 'differential_evolution',
                        initial_guess: Optional[Union[str, List[float], np.ndarray]] = None,
                        optimization_params: Optional[Dict] = None) -> Dict:
        """
        Optimize pricing to maximize/minimize target metric subject to constraints.
        
        Args:
            target: Target metric to optimize ('delta_bookings', 'delta_pmm', 'delta_rides')
            maximize: If True maximize, if False minimize the target
            constraints: List of constraint dictionaries with 'metric', 'operator', 'value'
                        e.g., [{'metric': 'delta_pmm', 'operator': '>=', 'value': -10000000}]
            price_bounds: Tuple of (min, max) price change in basis points
                         If None, will be set based on price_direction
            price_direction: 'increase', 'decrease', or 'both'
                           - 'increase': bounds [0, 20000] 
                           - 'decrease': bounds [-20000, 0]
                           - 'both': bounds [-20000, 20000] (or custom price_bounds)
            method: Optimization method ('differential_evolution', 'minimize', 'dual_annealing', 'basinhopping')
            initial_guess: Initial guess strategy:
                          - None: smart default based on price_direction
                          - 'random': random within bounds
                          - 'negative': small negative values (price decreases)
                          - 'positive': small positive values (price increases)
                          - 'large_positive': large positive values (10000bp)
                          - 'large_negative': large negative values (-10000bp)
                          - array: custom initial values
            optimization_params: Additional parameters for the optimizer
                                 For 'differential_evolution', can include 'workers': -1 for all CPUs.
        
        Returns:
            Dictionary with optimization results
        """
        if self.merged_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Set bounds based on direction if not provided
        if price_bounds is None:
            if price_direction == 'increase':
                price_bounds = (0, 20000)
            elif price_direction == 'decrease':
                price_bounds = (-20000, 0)
            elif price_direction == 'both':
                price_bounds = (-20000, 20000)
            else:
                raise ValueError("price_direction must be 'increase', 'decrease', or 'both'")
        
        n_products = len(self.merged_df)
        bounds = [price_bounds] * n_products
        
        # Set default optimization parameters
        default_params = {
            'differential_evolution': {
                'seed': 42,
                'maxiter': 2000,
                'atol': 1e-8,
                'tol': 1e-8,
                'popsize': 15,
                'mutation': (0.5, 1.5),
                'recombination': 0.7,
                'workers': -1,  # Default to use all CPU cores
            },
            'dual_annealing': {
                'maxiter': 2000,
                'initial_temp': 5230.0,
                'restart_temp_ratio': 2e-05,
                'visit': 2.62,
                'accept': -5.0,
                'seed': 42
            },
            'basinhopping': {
                'niter': 1000,
                'T': 1.0,
                'stepsize': 100.0,
                'seed': 42
            },
            'minimize': {
                'method': 'SLSQP',
                'options': {'maxiter': 1000, 'ftol': 1e-9}
            }
        }
        
        # Update with user parameters
        params = default_params.get(method, {})
        if optimization_params:
            params.update(optimization_params)
        
        # Define objective and constraint functions using the class methods
        objective_func = lambda price_changes: self._objective(price_changes, target, maximize)
        constraint_func = lambda price_changes: self._constraint_function(price_changes, constraints)

        # Set up constraints for scipy
        scipy_constraints = []
        if constraints:
            for i, constraint in enumerate(constraints):
                if constraint['operator'] in ['>=', '<=']:
                    scipy_constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=i: constraint_func(x)[idx]
                    })
                elif constraint['operator'] == '==':
                    scipy_constraints.append({
                        'type': 'eq',
                        'fun': lambda x, idx=i: constraint_func(x)[idx]
                    })
        
        # Generate initial guess
        x0 = self._generate_initial_guess(initial_guess, n_products, price_bounds, price_direction)
        
        print(f"Starting optimization with method: {method}")
        print(f"Price direction: {price_direction}")
        print(f"Initial guess strategy: {initial_guess}")
        print(f"Bounds: {price_bounds} basis points")
        print(f"Initial objective value: {objective_func(x0):.2f}")
        
        # Test constraint satisfaction with initial guess
        if constraints:
            initial_constraints = constraint_func(x0)
            print(f"Initial constraint values (diff from target): {initial_constraints}")
        
        result = None # Initialize result to None
        try:
            if method == 'differential_evolution':
                result = differential_evolution(
                    objective_func, 
                    bounds, 
                    constraints=scipy_constraints if scipy_constraints else None,
                    **params
                )
            
            elif method == 'dual_annealing':
                result = dual_annealing(
                    objective_func, 
                    bounds, 
                    **params
                )
                # Check constraints manually for dual_annealing
                if constraints and result.success:
                    constraint_vals = constraint_func(result.x)
                    # For dual_annealing, constraints are not handled by the solver directly.
                    # We check if the solution found violates them.
                    # A violation occurs if an inequality constraint (e.g., metric >= value) results
                    # in a negative difference (metric - value < 0), or an equality constraint
                    # is significantly non-zero. Allow for small numerical errors.
                    is_violated = False
                    for i, val in enumerate(constraint_vals):
                        op = constraints[i]['operator']
                        if (op in ['>=', '<='] and val < -1e-6) or (op == '==' and abs(val) > 1e-6):
                            is_violated = True
                            break
                    if is_violated:
                        result.success = False
                        result.message = "Solution violates constraints"
            
            elif method == 'basinhopping':
                # Note: basinhopping's minimizer_kwargs 'method' must support bounds and constraints.
                minimizer_method = params.pop('method', 'SLSQP') # Pop method from params to avoid passing twice
                minimizer_options = params.pop('options', {'maxiter': 1000, 'ftol': 1e-9})
                minimizer_kwargs = {
                    "bounds": bounds, 
                    "constraints": scipy_constraints, 
                    "method": minimizer_method,
                    "options": minimizer_options
                }
                result = basinhopping(
                    objective_func, 
                    x0, 
                    minimizer_kwargs=minimizer_kwargs,
                    **params
                )
            
            else:  # minimize
                result = minimize(
                    objective_func, 
                    x0, 
                    bounds=bounds, 
                    constraints=scipy_constraints,
                    **params
                )
            
        except Exception as e:
            raise RuntimeError(f"Optimization failed with error: {e}")
        
        # ALWAYS store the best price changes found and results, regardless of convergence success
        if result is not None:
            self.set_price_changes(result.x)
            
            print(f"Optimization finished. Success status: {result.success}")
            print(f"Message: {result.message if hasattr(result, 'message') else 'No specific message'}")
            print(f"Final objective value: {-result.fun if maximize else result.fun:.2f}")
            print(f"Average price change: {np.mean(result.x):.1f} basis points")
            print(f"Price change range: [{np.min(result.x):.1f}, {np.max(result.x):.1f}] basis points")
            
            # Store results
            optimization_results = {
                'success': result.success,
                'optimal_price_changes': result.x,
                'optimal_value': -result.fun if maximize else result.fun,
                'target_metric': target,
                'maximize': maximize,
                'constraints': constraints,
                'price_direction': price_direction,
                'price_bounds': price_bounds,
                'method': method,
                'initial_guess_strategy': initial_guess,
                'message': result.message if hasattr(result, 'message') else 'Optimization completed (or stopped)',
                'iterations': getattr(result, 'nit', getattr(result, 'nfev', 'Unknown')),
                'function_evaluations': getattr(result, 'nfev', 'Unknown')
            }
            
            # Add constraint satisfaction
            if constraints:
                constraint_values = {}
                # Re-evaluate constraints using the final optimal price changes
                final_constraint_vals = self._constraint_function(result.x, constraints)
                for i, constraint in enumerate(constraints):
                    # Store the actual difference from target for clarity
                    constraint_values[constraint['metric']] = final_constraint_vals[i] 
                optimization_results['constraint_values'] = constraint_values
                print(f"Final constraint values (diff from target): {constraint_values}")
            
            self.optimization_results = optimization_results
            return optimization_results
        else:
            raise RuntimeError(f"Optimization did not return any result. An unexpected error might have occurred.")
    
    def _generate_initial_guess(self, initial_guess, n_products, price_bounds, price_direction):
        """Generate initial guess based on strategy and price direction."""
        if initial_guess is None:
            # Smart default based on price direction
            if price_direction == 'increase':
                # Small positive values by default for 'increase' if None
                return self._generate_initial_guess('positive', n_products, price_bounds, price_direction)
            elif price_direction == 'decrease':
                # Small negative values by default for 'decrease' if None
                return self._generate_initial_guess('negative', n_products, price_bounds, price_direction)
            else:  # 'both'
                return np.zeros(n_products) # Start with no change for 'both' if None
        
        elif initial_guess == 'random':
            np.random.seed(42)
            return np.random.uniform(price_bounds[0], price_bounds[1], n_products)
        
        elif initial_guess == 'negative':
            # Small price decreases (around -0.5% to -1%)
            np.random.seed(42)
            # Ensure values are within bounds
            return np.random.uniform(max(-100, price_bounds[0]), min(-25, price_bounds[1]), n_products)
        
        elif initial_guess == 'positive':
            # Small price increases (around 0.25% to 1%)
            np.random.seed(42)
            # Ensure values are within bounds
            return np.random.uniform(max(25, price_bounds[0]), min(100, price_bounds[1]), n_products)
        
        elif initial_guess == 'large_positive':
            # Large price increases (around 100% = 10000bp), capped by upper bound
            return np.full(n_products, min(10000, price_bounds[1]))
        
        elif initial_guess == 'large_negative':
            # Large price decreases (around -100% = -10000bp), capped by lower bound
            return np.full(n_products, max(-10000, price_bounds[0]))
        
        elif isinstance(initial_guess, (list, np.ndarray)):
            if len(initial_guess) != n_products:
                raise ValueError(f"Custom initial guess length ({len(initial_guess)}) must match number of products ({n_products})")
            # Ensure custom initial guess values are within bounds
            for val in initial_guess:
                if not (price_bounds[0] <= val <= price_bounds[1]):
                    warnings.warn(f"Initial guess value {val} is outside price bounds {price_bounds}. It will be clamped.")
            return np.clip(initial_guess, price_bounds[0], price_bounds[1])
        
        else:
            raise ValueError(f"Unknown initial_guess strategy: {initial_guess}")
    
    def test_scenario(self, uniform_price_change: float) -> Dict:
        """
        Test a uniform price change scenario to validate optimization logic.
        
        Args:
            uniform_price_change: Price change in basis points to apply to all products
            
        Returns:
            Dictionary with scenario results
        """
        if self.merged_df is None:
            raise ValueError("No data loaded.")
        
        # Apply uniform price change
        original_changes = self.merged_df['price_change'].copy()
        self.set_price_changes([uniform_price_change] * len(self.merged_df))
        
        # Calculate metrics
        results = {
            'price_change_bps': uniform_price_change,
            'total_delta_bookings': self.merged_df['delta_bookings'].sum(),
            'total_delta_pmm': self.merged_df['delta_pmm'].sum(),
            'total_delta_rides': self.merged_df['delta_rides'].sum(),
            'avg_elasticity': self.merged_df['elasticity'].mean(),
            'rides_weighted_elasticity': (self.merged_df['elasticity'] * self.merged_df['rides']).sum() / self.merged_df['rides'].sum()
        }
        
        # Restore original price changes
        self.merged_df['price_change'] = original_changes
        self._update_derived_metrics()
        
        return results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get the results dataframe with all calculations and optimal price changes.
        
        Returns:
            DataFrame with all metrics and optimal price changes
        """
        if self.merged_df is None:
            raise ValueError("No data available. Load data and run optimization first.")
        
        self.results_df = self.merged_df.copy()
        return self.results_df
    
    def get_summary_metrics(self) -> Dict:
        """
        Get summary metrics for the current pricing scenario.
        
        Returns:
            Dictionary with total metrics
        """
        if self.merged_df is None:
            raise ValueError("No data available.")
        
        return {
            'total_delta_bookings': self.merged_df['delta_bookings'].sum(),
            'total_delta_pmm': self.merged_df['delta_pmm'].sum(),
            'total_delta_rides': self.merged_df['delta_rides'].sum(),
            'avg_price_change_bps': self.merged_df['price_change'].mean(),
            'n_products': len(self.merged_df)
        }
    
    def analyze_sensitivity(self, price_change_range: Tuple[float, float] = (-1000, 1000), 
                          steps: int = 50) -> pd.DataFrame:
        """
        Perform sensitivity analysis across a range of uniform price changes.
        
        Args:
            price_change_range: Range of price changes to test (in basis points)
            steps: Number of steps in the range
        
        Returns:
            DataFrame with sensitivity analysis results
        """
        if self.merged_df is None:
            raise ValueError("No data loaded.")
        
        price_changes = np.linspace(price_change_range[0], price_change_range[1], steps)
        results = []
        
        for pc in price_changes:
            # Set uniform price change for all products
            self.set_price_changes([pc] * len(self.merged_df))
            
            results.append({
                'price_change_bps': pc,
                'total_delta_bookings': self.merged_df['delta_bookings'].sum(),
                'total_delta_pmm': self.merged_df['delta_pmm'].sum(),
                'total_delta_rides': self.merged_df['delta_rides'].sum(),
                'avg_new_price': self.merged_df['new_price'].mean(),
                'total_new_13w_bookings': self.merged_df['new_13w_bookings'].sum()
            })
        
        return pd.DataFrame(results)


# Example usage and helper functions
def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample data for testing the optimizer with multiple records per region-product."""
    np.random.seed(42)
    
    # Create base region-product combinations
    n_base_products = 20
    regions = ['North', 'South', 'East', 'West']
    
    base_combinations = []
    for region in regions:
        for i in range(n_base_products):
            base_combinations.append({
                'region': region,
                'product': f'Product_{i:03d}'
            })
    
    # Now create detailed records with use_case and db variations
    use_cases = ['Type_A', 'Type_B', 'Type_C']
    dbs = ['Cat_1', 'Cat_2', 'Cat_3']
    
    history_records = []
    price_records = []
    
    for base in base_combinations:
        # Create 2-4 records per region-product combination with different type/db
        n_variations = np.random.randint(2, 5)
        
        for _ in range(n_variations):
            # History record
            rides = np.random.randint(50, 500)
            history_record = {
                'region': base['region'],
                'analytical_ride_type': base['product'], # Corrected from 'analytical_ride_type' to 'product' for sample data generation
                'use_case': np.random.choice(use_cases), 
                'db': np.random.choice(dbs),
                'pmm': np.random.randint(500, 5000),
                'bookings': np.random.randint(2000, 20000),
                'rides': rides,
                'elasticity': np.random.uniform(-4, -0.8)  # Different elasticity per variation
            }
            history_records.append(history_record)
            
            # Corresponding price record
            price_record = {
                'region': base['region'],
                'analytical_ride_type': base['product'], # Corrected here too
                'use_case': history_record['use_case'],  # Match the use_case/db
                'db': history_record['db'],
                'sessions': np.random.randint(20, 200),
                'total_price': np.random.randint(1000, 10000)
            }
            price_records.append(price_record)
    
    return pd.DataFrame(history_records), pd.DataFrame(price_records)


# Demonstration
if __name__ == "__main__":
    # Create sample data
    history_df, price_df = create_sample_data()
    
    print("=== Example 1: Region-analytical_ride_type Level Aggregation ===")
    # Initialize optimizer with no additional levels (region-analytical_ride_type only)
    optimizer1 = PriceOptimizer()
    optimizer1.load_data(history_df, price_df)
    
    print("Data loaded and aggregated successfully!")
    print(f"Original history records: {len(history_df)}")
    print(f"Original price records: {len(price_df)}")
    print(f"Aggregated to {len(optimizer1.merged_df)} region-analytical_ride_type combinations")
    print("\nSample of aggregated data:")
    print(optimizer1.merged_df[['region', 'analytical_ride_type', 'avg_price', 'weekly_rides', 'elasticity']].head())
    
    print("\n=== Example 2: Region-analytical_ride_type-use_case Level Aggregation ===")
    # Initialize optimizer with use_case level
    optimizer2 = PriceOptimizer(levels=['use_case'])
    optimizer2.load_data(history_df, price_df)
    
    print(f"Aggregated to {len(optimizer2.merged_df)} region-analytical_ride_type-use_case combinations")
    print("\nSample of aggregated data:")
    print(optimizer2.merged_df[['region', 'analytical_ride_type', 'use_case', 'avg_price', 'weekly_rides', 'elasticity']].head())
    
    print("\n=== Example 3: Region-analytical_ride_type-use_case-db Level Aggregation ===")
    # Initialize optimizer with both use_case and db levels
    optimizer3 = PriceOptimizer(levels=['use_case', 'db'])
    optimizer3.load_data(history_df, price_df)
    
    print(f"Aggregated to {len(optimizer3.merged_df)} region-analytical_ride_type-use_case-db combinations")
    print("\nSample of aggregated data:")
    print(optimizer3.merged_df[['region', 'analytical_ride_type', 'use_case', 'db', 'avg_price', 'weekly_rides', 'elasticity']].head())
    
    # Test different scenarios first
    print("\n=== Testing Different Scenarios ===")
    test_scenarios = [-100, -50, 0, 50, 100]  # basis points
    for scenario_pc in test_scenarios:
        scenario_results = optimizer1.test_scenario(scenario_pc)
        print(f"\nScenario (Uniform Price Change = {scenario_pc} bps):")
        print(f"  Total Delta Bookings: {scenario_results['total_delta_bookings']:.2f}")
        print(f"  Total Delta PMM: {scenario_results['total_delta_pmm']:.2f}")
        print(f"  Total Delta Rides: {scenario_results['total_delta_rides']:.2f}")

    print("\n=== Running Optimization Example ===")
    # Example: Maximize delta_pmm with a minimum delta_rides constraint
    constraints = [
        {'metric': 'delta_rides', 'operator': '>=', 'value': -10000}, # Allow up to 10,000 fewer rides
        {'metric': 'delta_bookings', 'operator': '>=', 'value': -500000} # Allow up to 500k fewer bookings
    ]

    try:
        # Using differential_evolution with default 'workers': -1 (all CPUs)
        opt_results = optimizer1.optimize_pricing(
            target='delta_pmm',
            maximize=True,
            constraints=constraints,
            price_direction='both',
            method='differential_evolution',
            initial_guess=None, # Let it choose smart default
            optimization_params={'maxiter': 500} # Reduced iterations for quick demo
        )
        print("\nOptimization Results:")
        print(opt_results)
        
        results_df = optimizer1.get_results_dataframe()
        print("\nSample of Optimized Results Dataframe:")
        print(results_df[['region', 'analytical_ride_type', 'avg_price', 'new_price', 'price_change', 
                          'delta_pmm', 'delta_bookings', 'delta_rides']].head())
        print("\nSummary Metrics after Optimization:")
        print(optimizer1.get_summary_metrics())

    except RuntimeError as e:
        print(f"\nOptimization failed for example: {e}")

    print("\n=== Running another Optimization Example (Minimize delta_rides) ===")
    # Example: Minimize delta_rides with a minimum delta_pmm constraint
    constraints_min_rides = [
        {'metric': 'delta_pmm', 'operator': '>=', 'value': 0} # PMM must not decrease
    ]
    try:
        opt_results_min_rides = optimizer2.optimize_pricing( # Using optimizer2 (use_case level)
            target='delta_rides',
            maximize=False, # Minimize rides
            constraints=constraints_min_rides,
            price_direction='increase', # Only allow price increases
            method='differential_evolution',
            optimization_params={'maxiter': 200, 'workers': 2} # Explicitly set workers for demo
        )
        print("\nOptimization Results (Minimize Delta Rides):")
        print(opt_results_min_rides)
        print("\nSummary Metrics after Minimizing Delta Rides Optimization:")
        print(optimizer2.get_summary_metrics())

    except RuntimeError as e:
        print(f"\nOptimization failed for minimize rides example: {e}")

    print("\n=== Sensitivity Analysis Example ===")
    sensitivity_df = optimizer1.analyze_sensitivity(price_change_range=(-500, 500), steps=20)
    print("\nSample of Sensitivity Analysis Results:")
    print(sensitivity_df.head())
    print("\nFull Sensitivity Analysis Dataframe Shape:", sensitivity_df.shape)