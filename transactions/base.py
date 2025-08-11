class Transaction(object):
    @classmethod
    def from_json(cls, json_list):
        tx_as_json = []
        for d in json_list:
            if 'sales' in d:
                tx_as_json.append(cls(d['product'], d['offered_products'], d['sales']))
            else:
                tx_as_json.append(cls(d['product'], d['offered_products'], 1))
        return tx_as_json

    def __init__(self, product, offered_products, sales=1):
        self.product = product
        self.offered_products = offered_products
        self.sales = sales

    def as_json(self):
        return {'product': self.product, 'offered_products': self.offered_products, 'sales': self.sales}

    def __hash__(self):
        return (self.product, tuple(self.offered_products)).__hash__()

    def __eq__(self, other):
        return self.product == other.product and self.offered_products == other.offered_products

    def __repr__(self):
        return "<Product: %s ; Offered products: %s >" % (self.product, self.offered_products)
