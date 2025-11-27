import json
import random
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from faker import Faker
import os
import re
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FAKE_ALL_PATH = PROJECT_ROOT /'data/raw/fake_all.json'
from platform_specs import PLATFORM_SPECS

fake = Faker("zh_CN")

def gen_time_str(base_dt, precision):
    if precision == "hm":
        return base_dt.strftime("%Y-%m-%d %H:%M")
    else:
        return base_dt.strftime("%Y-%m-%d %H:%M:%S")

def random_amount(min_v, max_v):
    """生成两位小数金额"""
    value = Decimal(str(random.uniform(min_v, max_v)))
    return value.quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)

def inject_special_char(addr: str) -> str:
    """可选往地址里插入｜模拟脏数据"""
    if len(addr) < 4:
        return addr
    pos = random.randint(2, len(addr)-1)
    return addr[:pos] + "｜" + addr[pos:]

def gen_cn_mobile():
    """生成一个看起来像中国大陆手机号的串"""
    # 以 1 开头的 11 位数字
    second = random.choice(["3", "5", "7", "8", "9"])
    rest = "".join(random.choice("0123456789") for _ in range(9))
    return "1" + second + rest


# def gen_trips_for_platform(platform: str, cfg: dict, city: str):
#     trips = []
#     n = random.randint(1, 4)

#     # 平台可选配置（没有就用默认）
#     speed_min, speed_max = cfg.get("speed_kmh", (22, 34))         # 合理城市平均车速
#     base_min, base_max = cfg.get("pricing", {}).get("base", (8, 16))
#     per_km_val = cfg.get("pricing", {}).get("per_km", 2.4)        # 元/公里
#     per_km = Decimal(str(per_km_val))

#     for _ in range(n):
#         start = fake.date_time_between(start_date="-30d", end_date="now")
#         duration_min = random.randint(8, 40)
#         end = start + timedelta(minutes=duration_min)
#         weekday_cn = ["周一","周二","周三","周四","周五","周六","周日"][start.weekday()]
#         # 在同一个 city 下生成起终点
#         origin_detail = fake.street_address()
#         dest_detail = fake.street_address()
#         origin = f"{city}{origin_detail}"
#         destination = f"{city}{dest_detail}"

#         # 随机插入特殊字符模拟脏数据
#         if random.random() < 0.3:
#             origin = inject_special_char(origin)
#         if random.random() < 0.1:
#             destination = inject_special_char(destination)

#         # —— 里程（公里）——
#         avg_speed = random.uniform(speed_min, speed_max)                   # km/h
#         distance = avg_speed * (duration_min / 60.0) + random.uniform(-0.3, 0.3)
#         # 合理边界并四舍五入到 0.01 km
#         distance = max(1.5, min(60.0, distance))
#         distance_km = Decimal(str(distance)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

#         # —— 金额：起步价 + 里程费 ——（保证与里程一致，无“幻觉”）
#         base_fare = random_amount(base_min, base_max)                      # Decimal，保留两位
#         amount_dec = (base_fare + per_km * distance_km).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
#         invoice_amount = float(amount_dec)

#         trips.append({
#             "start_time_str": gen_time_str(start, cfg["time_precision"]),
#             "end_time_str": gen_time_str(end, cfg["time_precision"]),
#             "weekday_cn": weekday_cn,
#             "service_provider": cfg["service_provider"],
#             "car_type": random.choice(["快车", "专车", "商务"]),
#             "city": city,
#             "origin": origin,
#             "destination": destination,
#             "distance_km": float(distance_km),        # 👈 新增：里程（公里，保留两位）
#             "invoice_amount": invoice_amount
#         })

#     return trips
def gen_trips_for_platform(platform: str, cfg: dict, city: str):
    trips = []
    n = random.randint(1, 10)

    speed_min, speed_max = cfg.get("speed_kmh", (22, 34))
    base_min, base_max = cfg.get("pricing", {}).get("base", (8, 16))
    per_km_val = cfg.get("pricing", {}).get("per_km", 2.4)
    per_km = Decimal(str(per_km_val))

    extra_min, extra_max = cfg.get("pricing", {}).get("extra_fee", (0, 5))
    disc_min,  disc_max  = cfg.get("pricing", {}).get("discount",  (0, 8))

    for _ in range(n):
        start = fake.date_time_between(start_date="-30d", end_date="now")
        duration_min = random.randint(8, 40)
        end = start + timedelta(minutes=duration_min)
        weekday_cn = ["周一","周二","周三","周四","周五","周六","周日"][start.weekday()]

        origin_detail = fake.street_address()
        dest_detail   = fake.street_address()
        origin = f"{city}{origin_detail}"
        destination = f"{city}{dest_detail}"

        if random.random() < 0.3:
            origin = inject_special_char(origin)
        if random.random() < 0.1:
            destination = inject_special_char(destination)

        avg_speed = random.uniform(speed_min, speed_max)
        distance  = avg_speed * (duration_min / 60.0) + random.uniform(-0.3, 0.3)
        distance  = max(1.5, min(60.0, distance))
        distance_km = Decimal(str(distance)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        base_fare  = random_amount(base_min, base_max)
        trip_fee   = (base_fare + per_km * distance_km).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)

        extra_fee  = random_amount(extra_min, extra_max)

        cap = (trip_fee + extra_fee).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
        max_discount = max(Decimal("0.00"), cap - Decimal("0.01"))
        raw_discount = random_amount(disc_min, disc_max)
        discount    = min(raw_discount, max_discount).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)

        order_total = (trip_fee + extra_fee - discount).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
        invoice_amount = float(order_total)

        trip_obj = {
            "start_time_str": gen_time_str(start, cfg["time_precision"]),
            "end_time_str":   gen_time_str(end,   cfg["time_precision"]),
            "weekday_cn":     weekday_cn,
            "service_provider": cfg["service_provider"],
            "car_type":         random.choice(["快车", "专车", "商务"]),
            "city": city,
            "origin": origin,
            "destination": destination,
            "distance_km": float(distance_km),

            "trip_fee":     float(trip_fee),
            "extra_fee":    float(extra_fee),
            "discount":     float(discount),
            "order_total":  float(order_total),
            "invoice_amount": invoice_amount,

            # ✅ 临时塞一个排序用的 datetime
            "_start_dt": start,
        }
        trips.append(trip_obj)

    if random.random() < 0.8:
        trips.sort(key=lambda x: x["_start_dt"])  # 80% 有序
    else:
        random.shuffle(trips)                    # 20% 乱序

    return trips

def gen_receipt(platform: str, seq: int):
    cfg = PLATFORM_SPECS[platform]
    sample_id = f"{platform}_{seq:06d}"

    # 👇 这一张单用同一个城市
    city = fake.city_name()
    trips = gen_trips_for_platform(platform, cfg, city)
    trip_num = len(trips)

    total_amount = float(
        Decimal(str(sum(t["invoice_amount"] for t in trips))).quantize(
            Decimal("0.00"), rounding=ROUND_HALF_UP
        )
    )
    start_dts = [t["_start_dt"] for t in trips]
    first_trip_date = min(start_dts).strftime("%Y-%m-%d")
    last_trip_date  = max(start_dts).strftime("%Y-%m-%d")

    # 算完再清理掉内部字段，免得进 fake_all.json
    for t in trips:
        t.pop("_start_dt", None)
    apply_dt = datetime.strptime(last_trip_date, "%Y-%m-%d") + timedelta(
        days=random.randint(1, 7)
    )

    receipt = {
        "id": sample_id,
        "platform": platform,                 # 👈 存内部ID，比如 "baidu"
        "platform_name": cfg["platform_name"],  # 中文名
        "order_id": f"TX{random.randint(10**9, 10**10 - 1)}",
        "trip_num":trip_num,
        "passenger_name": fake.name(),
        "passenger_phone": gen_cn_mobile(),
        "first_trip_date": first_trip_date,
        "trip_date": last_trip_date,
        "apply_date": apply_dt.strftime("%Y-%m-%d"),
        "total_amount": total_amount,
        "currency": "CNY",
        "trips": trips,
    }
    return receipt

def get_next_index_for_platform(platform: str) -> int:
    """
    扫描 fake_all.json 里所有该平台的记录，找到 id 类似
    baidu_000123 的最大编号，返回 max+1。
    如果没有记录，则从 1 开始。
    """
    if not os.path.exists(FAKE_ALL_PATH):
        # 没有文件，说明还没生成过，直接从 1 开始
        return 1

    with open(FAKE_ALL_PATH, "r", encoding="utf-8") as f:
        receipts = json.load(f)

    pattern = re.compile(rf"^{re.escape(platform)}_(\d{{6}})$")
    max_idx = 0

    for r in receipts:
        if r.get("platform") != platform:
            continue
        rid = r.get("id") or ""
        m = pattern.match(rid)
        if not m:
            continue
        idx = int(m.group(1))
        max_idx = max(max_idx, idx)

    # 如果没找到任何匹配记录，就从 1 开始
    return max_idx + 1 if max_idx > 0 else 1


# def generate_all():
#     all_data = []
#     seq_map = {p: 1 for p in PLATFORM_SPECS.keys()}
    
    
#     for platform in PLATFORM_SPECS.keys():
#         for _ in range(100):
#             r = gen_receipt(platform, seq_map[platform])
#             seq_map[platform] += 1
#             all_data.append(r)

#     os.makedirs("data/raw", exist_ok=True)
#     with open("data/raw/fake_all.json", "w", encoding="utf-8") as f:
#         json.dump(all_data, f, ensure_ascii=False, indent=2)
#     print("生成完成：data/raw/fake_all.json")

def generate_all(num_per_platform: int = 10):
    # 1) 先把旧数据读进来（如果有的话）
    if os.path.exists(FAKE_ALL_PATH):
        with open(FAKE_ALL_PATH, "r", encoding="utf-8") as f:
            all_data = json.load(f)   # 旧的所有样本
        print(f"[INFO] 已加载历史数据：{len(all_data)} 条")
    else:
        all_data = []
        print("[INFO] 未找到历史数据，从空列表开始")

    # 2) 计算每个平台的起始编号（用你写好的 get_next_index_for_platform）
    seq_map = {}
    for p in PLATFORM_SPECS.keys():
        start_idx = get_next_index_for_platform(p)   # 已经是 max+1
        seq_map[p] = start_idx
        print(f"[{p}] 下一个起始编号: {start_idx:06d}")

    # 3) 生成本次新增的数据
    new_data = []
    for platform in PLATFORM_SPECS.keys():
        for _ in range(num_per_platform):
            r = gen_receipt(platform, seq_map[platform])
            seq_map[platform] += 1
            new_data.append(r)

    print(f"[INFO] 本次新生成：{len(new_data)} 条")

    # 4) 旧数据 + 新数据 合并后写回 fake_all.json
    all_data.extend(new_data)

    os.makedirs("data/raw", exist_ok=True)
    with open(FAKE_ALL_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"生成完成：{FAKE_ALL_PATH}，当前总计 {len(all_data)} 条")

if __name__ == "__main__":
    generate_all()