import pandas as pd
claims = pd.read_csv("data/claims.csv", keep_default_na=False)
gt = pd.read_csv("data/ground_truth.csv")

bad = (claims.claim_type.isin(("Co-op Marketing", "Promo Discount Support"))
       & (claims.claimed_amount_usd > 2500))
claims.loc[bad, "claim_type"] = "Listing Fee"
claims.loc[bad, "promo_id"] = ""
claims.loc[bad, "Raw_Claim_Text"] = [f"Annual listing fee settlement. USD {a:,.2f}. per trade terms"
                                     for a in claims.loc[bad, "claimed_amount_usd"]]

exp = gt.defect_type == "expired_window"
gt.loc[exp, "overclaim_usd"] = gt.loc[exp, "claimed_amount_usd"]
gt.loc[exp, "entitled_amount_usd"] = 0.0

claims.to_csv("data/claims.csv", index=False)
gt.to_csv("data/ground_truth.csv", index=False)
print(f"fixed {int(bad.sum())} claims, {int(exp.sum())} truth rows | leakage now US${gt.overclaim_usd.sum():,.0f}")