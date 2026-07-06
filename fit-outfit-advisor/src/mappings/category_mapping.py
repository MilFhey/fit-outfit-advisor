CATEGORY_MAPPING = {
    # Canonical Fashion V1 outputs
    "top": "top",
    "bottom": "bottom",
    "dress": "dress",
    "shoes": "shoes",
    "outerwear": "outerwear",
    "bag": "bag",
    "accessory": "accessory",

    # Fashion V1 product_type_v0 outputs
    "tshirt": "top",
    "shirt": "top",
    "jeans": "bottom",
    "trousers": "bottom",
    "shorts": "bottom",
    "casual_shoes": "shoes",
    "sports_shoes": "shoes",
    "dress_shoes": "shoes",
    "watch": "accessory",
    "sunglasses": "accessory",
    "cap": "accessory",

    # Tops
    "Shirts": "top",
    "Tshirts": "top",
    "Tops": "top",
    "Kurtas": "top",
    "Sweatshirts": "top",

    # Bottoms
    "Jeans": "bottom",
    "Trousers": "bottom",
    "Shorts": "bottom",
    "Track Pants": "bottom",
    "Skirts": "bottom",

    # Dresses / one-piece
    "Dresses": "dress",
    "Jumpsuit": "dress",

    # Shoes
    "Casual Shoes": "shoes",
    "Sports Shoes": "shoes",
    "Formal Shoes": "shoes",
    "Flats": "shoes",
    "Heels": "shoes",
    "Sandals": "shoes",

    # Accessories
    "Bags": "accessory",
    "Watches": "accessory",
    "Belts": "accessory",
    "Sunglasses": "accessory",
}


def map_to_common_category(raw_category: str) -> str:
    """Convertit une classe brute du dataset image vers une catégorie commune."""
    if not raw_category:
        return "unknown"
    return CATEGORY_MAPPING.get(raw_category, "unknown")
