"""Constants module for the scraper project."""

# XPath mappings for MercadoLibre website
XPATH_MAPPINGS = {
    "product_list": "//section/ol/li",
    "description": "//section/ol/li[{}]/div/div/div[2]/h3",
    "brand_options": [
        "//section/ol/li[{}]/div/div/div[2]/span[1]",
        "//section/ol/li[{}]/div/div/div[2]/span",
        "//section/ol/li[{}]/div/div/div[2]/span[2]",
    ],
    "original_price": "//section/ol/li[{}]/div/div/div[2]/div[2]/s",
    "offer_price_options": [
        "//section/ol/li[{}]/div/div/div[2]/div[2]/div/span[1]",
        "//section/ol/li[{}]/div/div/div[2]/div[3]/div/span[1]",
    ],
}
