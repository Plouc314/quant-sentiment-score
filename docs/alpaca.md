# Alpaca Market Data API

[Alpaca](https://alpaca.markets) is a US-based brokerage and market data provider. This document covers the market data API used for fetching historical OHLCV bars for US equities.

Base URL: `https://data.alpaca.markets`

---

## Authentication

Every request must include two HTTP headers carrying your API credentials:

| Header | Value |
|--------|-------|
| `APCA-API-KEY-ID` | Your Alpaca API key |
| `APCA-API-SECRET-KEY` | Your Alpaca API secret |

Credentials are obtained from the Alpaca dashboard. There is no OAuth flow — the keys are long-lived and sent with every request.

**Error responses:**
- `401 Unauthorized` — key is missing or malformed
- `403 Forbidden` — key is valid but lacks permission for the requested resource (e.g. SIP feed on a free plan)

---

## Endpoints

### GET `/v2/stocks/bars`

Returns historical OHLCV bars for one or more stock symbols.

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbols` | string | yes | Comma-separated list of ticker symbols (e.g. `AAPL,MSFT`) |
| `timeframe` | string | yes | Aggregation interval — see [Timeframes](#timeframes) |
| `start` | string (RFC3339) | yes | Inclusive start of the time range (e.g. `2024-01-01T00:00:00Z`) |
| `end` | string (RFC3339) | yes | Inclusive end of the time range |
| `limit` | integer | no | Maximum number of bars to return per page. Omit to use the API default |
| `adjustment` | string | no | Corporate action adjustment — see [Adjustments](#adjustments). Default: `raw` |
| `feed` | string | no | Data feed — see [Data Feeds](#data-feeds) |
| `page_token` | string | no | Opaque token returned by a previous response to fetch the next page |

#### Response

```json
{
  "bars": {
    "AAPL": [
      {
        "t": "2024-01-02T05:00:00Z",
        "o": 185.32,
        "h": 186.74,
        "l": 183.92,
        "c": 185.85,
        "v": 53112345,
        "n": 412873,
        "vw": 185.21
      }
    ]
  },
  "next_page_token": "eyJh..."
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `bars` | object | Map of ticker symbol → array of bars |
| `next_page_token` | string | Present when more pages are available; pass as `page_token` in the next request. Absent or empty string when on the last page |

**Bar fields:**

| Field | JSON key | Type | Description |
|-------|----------|------|-------------|
| Timestamp | `t` | string (RFC3339) | Start time of the bar |
| Open | `o` | float64 | First trade price in the period |
| High | `h` | float64 | Highest trade price in the period |
| Low | `l` | float64 | Lowest trade price in the period |
| Close | `c` | float64 | Last trade price in the period |
| Volume | `v` | integer | Total number of shares traded |
| Trade count | `n` | integer | Number of individual trades executed |
| VWAP | `vw` | float64 | Volume-weighted average price (optional, may be absent) |

Prices are in USD. The timestamp marks the **start** of the bar's time window.

#### Pagination

When the response contains `next_page_token`, repeat the request with that token in the `page_token` parameter to retrieve the next batch. Continue until `next_page_token` is absent or empty.

---

## Timeframes

The `timeframe` parameter controls bar aggregation. Supported values:

| Value | Duration |
|-------|----------|
| `1Min` | 1 minute |
| `5Min` | 5 minutes |
| `15Min` | 15 minutes |
| `30Min` | 30 minutes |
| `1Hour` | 1 hour |
| `4Hour` | 4 hours |
| `1Day` | 1 calendar day |
| `1Week` | 1 week |
| `1Month` | 1 calendar month |

---

## Data Feeds

The `feed` parameter selects the data source:

| Value | Description |
|-------|-------------|
| `iex` | IEX exchange data — available on all plans, includes a subset of trades |
| `sip` | Securities Information Processor — consolidated tape from all US exchanges, requires a paid subscription |

If omitted, the account's default feed is used.

---

## Adjustments

The `adjustment` parameter controls how corporate actions (splits, dividends) affect historical prices:

| Value | Description |
|-------|-------------|
| `raw` | No adjustment — prices as reported at trade time (default) |
| `split` | Prices adjusted for stock splits only |
| `dividend` | Prices adjusted for dividends only |
| `all` | Prices adjusted for both splits and dividends |

---

## Rate Limits and Daily Limits

Alpaca enforces per-account rate limits that depend on the subscription plan:

- **Free plan**: limited to the IEX feed; rate limits are lower.
- **Paid plans (Algo Trader Plus / Unlimited)**: access to the SIP feed; higher (or no) rate limits.

When a limit is exceeded the API responds with `429 Too Many Requests`. The response may include a `Retry-After` header (integer seconds or HTTP-date) indicating when the client may retry.

There is no publicly documented fixed daily request cap in the API itself — limits are applied as request-per-minute or request-per-second quotas that vary by plan. Check the [Alpaca pricing page](https://alpaca.markets/data) for current plan limits.

**Client-side rate limiting** can be configured to stay under the quota by spacing requests evenly (token bucket, burst size 1). Set the `rate_limit` configuration to the desired maximum requests per second.

### HTTP Status Codes Summary

| Status | Meaning |
|--------|---------|
| `200 OK` | Success |
| `400 Bad Request` | Invalid parameters (e.g. malformed date, unsupported timeframe) |
| `401 Unauthorized` / `403 Forbidden` | Authentication failure or insufficient permissions |
| `404 Not Found` | Symbol not found or no data available for the requested range |
| `429 Too Many Requests` | Rate limit exceeded; respect `Retry-After` before retrying |
| `502 / 503 / 504` | Server-side error; safe to retry after a short wait |
