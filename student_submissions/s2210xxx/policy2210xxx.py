from policy import Policy
import numpy as np
from scipy.optimize import linprog

#Width is the number of elements with -1 in one column, which means that the width is vertical, axis = 1
#height is the number of elements with -1 in one row, which means that the height is horizontal, axis = 0
class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            #print("Products information before sorting: ", observation["products"])
            
            # Sort the products by size in descending order.
            # sorted() function is a built-in function for sorting. For more information see the documentation.
            products = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
            #print("Products information after sorting: ", products)
            
            #Take out the information about available stocks
            stocks = observation["stocks"]
            
            # Iterate through the products
            for product in products:
                
                # If the quantity of current product is greater than 0,
                # continue to insert it into the stocks
                if product["quantity"] > 0:
                    
                    # Get the size of the current product
                    prod_size = product["size"]
                    
                    # Iterate through all available stocks.
                    # Iterate using enumerate() to get both the index and the stock,
                    # See the document for information.
                    for stock_idx, stock in enumerate(stocks):
                        
                        # Get the size of the current stock
                        stock_w, stock_h = self._get_stock_size_(stock)
                        
                        # Get the size of the current product into two variables
                        prod_w, prod_h = prod_size

                        # Initial condition: The size of the product must be less than or equal to the size of the stock.
                        if stock_w >= prod_w and stock_h >= prod_h:
                            
                            # Iterate through all possible indexes in the x coordinate where we can put 
                            # the top-left corner of the product.
                            for x in range(stock_w - prod_w + 1):
                                
                                # Inner loop to iterate through all possible indexes in the y coordinate where we can put
                                # the bootom-right corner of the product.
                                for y in range(stock_h - prod_h + 1):
                                    
                                    # Check if the product can be placed at the current position
                                    # If yes, return the information for the environment to place the product.
                                    if self._can_place_(stock, (x, y), prod_size):
                                        
                                        # Necessary datas for the environment to place the product,
                                        # including: the index of the stock that we insert the new product,
                                        # the size of the product, and the position of the top-left corner of the product.
                                        return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}

                        # else, rotate the product and check if it can fit into the stock
                        if stock_w >= prod_h and stock_h >= prod_w:
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    
                                    # The slicing notation [::-1] is used to reverse the order of the elements in the list,
                                    # which means that the width and height of the product size are swapped.
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (x, y)}

            # If no valid position is found, return a dummy action
            return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}
        
        elif self.policy_id == 2:
            pass

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, size):
        # Check if the product can be placed at the given position
        
        # Get the x and y coordinates of the top-left corner of the product
        pos_x, pos_y = position
        
        # Get the width and height of the product
        width, height = size
        
        # As the "stocks" key of the observation is a list of 2D numpy arrays,
        # we check using the dimension of the stock and the position of the product
        if pos_x + width <= stock.shape[0] and pos_y + height <= stock.shape[1]:
            if np.all(stock[pos_x:pos_x + width, pos_y:pos_y + height] == -1):
                return True
        return False

    def calculate_available_space(self, stock):
        # Count the number of -1 elements in the stock
        available_space = np.sum(stock == -1)
        return available_space
    