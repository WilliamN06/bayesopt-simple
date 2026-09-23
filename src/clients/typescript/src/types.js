/**
 * Configuration options for the RateLimiter client
 */
export interface RateLimiterConfig {
  /** Base URL of the rate limiter service (e.g., 'http://localhost:8080') */
  host: string;
  
  /** Request timeout in milliseconds (default: 5000) */
  timeout?: number;
  
  /** Number of retry attempts for failed requests (default: 1) */
  retryCount?: number;
  
  /** Delay between retries in milliseconds (default: 100) */
  retryDelay?: number;
  
  /** Optional API key for authentication */
  apiKey?: string;
  
  /** Custom fetch implementation (for testing or polyfills) */
  fetch?: typeof fetch;
}

/**
 * Parameters for a rate limit check
 */
export interface CheckParams {
  /** Unique client identifier (e.g., API key, user ID) */
  clientId: string;
  
  /** Endpoint being accessed (e.g., '/api/users') */
  endpoint: string;
  
  /** Number of tokens to consume (default: 1) */
  quantity?: number;
  
  /** Override the algorithm for this check */
  algorithm?: 'fixed_window' | 'sliding_window' | 'token_bucket';
}

/**
 * Result of a rate limit check
 */
export interface CheckResult {
  /** Whether the request is allowed */
  allowed: boolean;
  
  /** Maximum number of requests allowed in the window */
  limit: number;
  
  /** Number of requests remaining in the current window */
  remaining: number;
  
  /** When the current window resets */
  resetAt: Date;
  
  /** Seconds to wait before retrying (only set when allowed is false) */
  retryAfter?: number;
}

/**
 * Configuration for the Express middleware
 */
export interface MiddlewareOptions {
  /** Function to extract the client ID from the request */
  extractClientId: (req: any) => string | undefined;
  
  /** Function to extract the endpoint from the request (default: req.path) */
  extractEndpoint?: (req: any) => string;
  
  /** Behavior when the rate limiter service is unavailable (default: true) */
  failOpen?: boolean;
  
  /** Custom handler for rate-limited requests */
  onRateLimited?: (req: any, res: any, result: CheckResult) => void;
  
  /** Custom handler for errors */
  onError?: (req: any, res: any, error: Error) => void;
}

/**
 * Configuration response from the server
 */
export interface ConfigResponse {
  defaultLimit: number;
  defaultWindow: string;
  defaultAlgorithm: string;
  supportedAlgorithms: string[];
  overrides?: Array<{
    clientId: string;
    limit: number;
    algorithm: string;
    window: string;
  }>;
}

/**
 * Error response from the server
 */
export interface ErrorResponse {
  error: string;
  code: string;
  details?: string;
}