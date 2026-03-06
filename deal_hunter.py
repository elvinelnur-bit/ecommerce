import warnings
warnings.filterwarnings("ignore")

import json
import google.generativeai as genai


# ===============================
# GEMINI API (shared with app.py for AI recommendations)
# ===============================
API_KEY = "AIzaSyC2EdQQ6WPsRS1oI_KTEzXutseHXp-4ADU"

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-pro")


# ===============================
# DISCOUNT FUNKSIYASI
# ===============================
def calculate_discount(old_price, new_price):
    discount = ((old_price - new_price) / old_price) * 100
    return discount


# ===============================
# MAIN (only run when executed as script)
# ===============================
if __name__ == "__main__":
    # DATA OXUMA (JSONL)
    stores_data = {}
    with open("data/stores.jsonl", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            store = item["store"]
            product = item["product"]
            old_price = item["old_price"]
            new_price = item["new_price"]
            rating = item["rating"]
            if store not in stores_data:
                stores_data[store] = {}
            stores_data[store][product] = {
                "old_price": old_price,
                "new_price": new_price,
                "rating": rating,
            }

    print("Market datası yükləndi\n")

    # USER INPUT
    shopping_input = input("Məhsulları vergüllə yaz: ")
    shopping_list = [p.strip().lower() for p in shopping_input.split(",")]
    print("\nAlış siyahısı:", shopping_list)

    # ƏN UCUZ MAĞAZANI TAP
    total_price = 0
    best_plan = {}
    super_deals = []

    for product in shopping_list:
        cheapest_store = None
        cheapest_price = float("inf")
        cheapest_old_price = None
        rating = None
        for store, products in stores_data.items():
            if product in products:
                data = products[product]
                price = data["new_price"]
                old_price = data["old_price"]
                rating = data["rating"]
                if price < cheapest_price:
                    cheapest_price = price
                    cheapest_store = store
                    cheapest_old_price = old_price

        if cheapest_store is None:
            print(product, "tapılmadı")
            continue

        discount = calculate_discount(cheapest_old_price, cheapest_price)
        total_price += cheapest_price
        if cheapest_store not in best_plan:
            best_plan[cheapest_store] = []
        best_plan[cheapest_store].append((product, cheapest_price))

        # SUPER DEAL RULE
        if discount >= 40 and rating > 4:
            print(f"🔥 SUPER DEAL TAPILDI → {product}")
            prompt = f"""
            Write a short catchy Instagram advertisement text
            for the product {product}.
            The product has a {round(discount,2)}% discount.
            """
            try:
                response = model.generate_content(prompt)
                ai_text = response.text
            except Exception:
                print("AI API işləmədi, default text istifadə olunur")
                ai_text = "Limited-time deal! Grab this product now before the discount ends!"

            super_deals.append({
                "product": product,
                "store": cheapest_store,
                "old_price": cheapest_old_price,
                "new_price": cheapest_price,
                "discount": round(discount, 2),
                "rating": rating,
                "ai_marketing_text": ai_text,
            })

    # NƏTİCƏ
    print("\n🛒 ƏN SƏRFƏLİ ALIŞ PLANI\n")
    for store, items in best_plan.items():
        print(store)
        for name, price in items:
            print(" ", name, "-", price, "AZN")
        print()
    print("Ümumi məbləğ:", round(total_price, 2), "AZN")

    # JSON PIPELINE
    with open("data/best_deals.json", "w", encoding="utf-8") as f:
        json.dump(super_deals, f, indent=4, ensure_ascii=False)
    print("\nSuper deals data/best_deals.json faylına yazıldı")