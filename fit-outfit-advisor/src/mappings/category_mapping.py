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
    "sandals": "shoes",
    "flip_flops": "shoes",
    "heels": "shoes",
    "flats": "shoes",
    "watch": "accessory",
    "sunglasses": "accessory",
    "cap": "accessory",
    "wallet": "accessory",
    "belt": "accessory",
    "jewellery": "accessory",

    # Tops
    "Shirts": "top",
    "Tshirts": "top",
    "Tops": "top",
    "Kurtas": "top",

    # Bottoms
    "Jeans": "bottom",
    "Trousers": "bottom",
    "Shorts": "bottom",
    "Track Pants": "bottom",
    "Skirts": "bottom",

    # Dresses / one-piece
    "Dresses": "dress",
    "Jumpsuit": "dress",

    # Outerwear
    "Sweatshirts": "outerwear",
    "Sweaters": "outerwear",
    "Jackets": "outerwear",

    # Shoes
    "Casual Shoes": "shoes",
    "Sports Shoes": "shoes",
    "Formal Shoes": "shoes",
    "Flats": "shoes",
    "Heels": "shoes",
    "Sandals": "shoes",
    "Flip Flops": "shoes",

    # Accessories
    "Bags": "bag",
    "Handbags": "bag",
    "Backpacks": "bag",
    "Clutches": "bag",
    "Watches": "accessory",
    "Belts": "accessory",
    "Sunglasses": "accessory",
    "Caps": "accessory",
    "Wallets": "accessory",
    "Earrings": "accessory",
    "Pendant": "accessory",
    "Necklace and Chains": "accessory",
}


def map_to_common_category(raw_category: str) -> str:
    """Convertit une classe brute du dataset image vers une catégorie commune."""
    if not raw_category:
        return "unknown"
    return CATEGORY_MAPPING.get(raw_category, "unknown")
