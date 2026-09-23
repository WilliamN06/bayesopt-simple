import {
  RateLimiterConfig,
  CheckParams,
  CheckResult,
  ConfigResponse,
  ErrorResponse,
} from './types';
import {
  ConnectionError,
  TimeoutError,
  ServerError,
  ValidationError,
} from './errors';

/**
 * Default configuration values
 */
const DEFAULT_CONFIG: Required<Omit<RateLimiterConfig, 'host' | 'apiKey' | 'fetch'>> = {
  timeout: 5000,
  retryCount: 1,
  retryDelay: 100,
};

/**
 * Client for the Rate Limiter service
 * 
 * @example
 * ```typescript
 * const limiter = new RateLimiter({ host: 'http://localhost:8080' });
 * const result = await limiter.check({ clientId: 'user-123', endpoint: '/api/users' });
 * if (!result.allowed) {
 *   console.log(`Rate limited. Retry after ${result.retryAfter} seconds`);
 * }
 * ```
 */
export class RateLimiter {
  private readonly host: string;
  private readonly timeout: number;
  private readonly retryCount: number;
  private readonly retryDelay: number;
  private readonly apiKey?: string;
  private readonly fetchFn: typeof fetch;

  constructor(config: RateLimiterConfig) {
    if (!config.host) {
      throw new ValidationError('host is required', 'host');
    }

    // Remove trailing slash
    this.host = config.host.replace(/\/$/, '');
    this.timeout = config.timeout ?? DEFAULT_CONFIG.timeout;
    this.retryCount = config.retryCount ?? DEFAULT_CONFIG.retryCount;
    this.retryDelay = config.retryDelay ?? DEFAULT_CONFIG.retryDelay;
    this.apiKey = config.apiKey;
    this.fetchFn = config.fetch ?? globalThis.fetch;

    if (!this.fetchFn) {
      throw new ValidationError(
        'fetch is not available. Provide a fetch implementation or use Node.js 18+',
        'fetch'
      );
    }
  }

  /**
   * Check if a request is allowed
   */
  async check(params: CheckParams): Promise<CheckResult> {
    // Validate parameters
    if (!params.clientId) {
      throw new ValidationError('clientId is required', 'clientId');
    }
    if (!params.endpoint) {
      throw new ValidationError('endpoint is required', 'endpoint');
    }

    const response = await this.request<{
      allowed: boolean;
      limit: number;
      remaining: number;
      resetAt: number;
      retryAfter?: number;
    }>('POST', '/v1/check', {
      clientId: params.clientId,
      endpoint: params.endpoint,
      quantity: params.quantity ?? 1,
      algorithm: params.algorithm,
    });

    return {
      allowed: response.allowed,
      limit: response.limit,
      remaining: response.remaining,
      resetAt: new Date(response.resetAt * 1000),
      retryAfter: response.retryAfter,
    };
  }

  /**
   * Get the current configuration from the server
   */
  async getConfig(): Promise<ConfigResponse> {
    return this.request<ConfigResponse>('GET', '/v1/config');
  }

  /**
   * Check if the service is healthy
   */
  async health(): Promise<boolean> {
    try {
      await this.request<{ status: string }>('GET', '/health');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Make an HTTP request with retry logic
   */
  private async request<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this.retryCount; attempt++) {
      if (attempt > 0) {
        await this.sleep(this.retryDelay * attempt);
      }

      try {
        return await this.makeRequest<T>(method, path, body);
      } catch (error) {
        lastError = error as Error;

        // Don't retry on validation errors or 4xx responses
        if (
          error instanceof ValidationError ||
          (error instanceof ServerError && error.statusCode < 500)
        ) {
          throw error;
        }
      }
    }

    throw lastError ?? new ConnectionError('Request failed after retries');
  }

  /**
   * Make a single HTTP request
   */
  private async makeRequest<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const headers: Record<string, string> = {
        'Accept': 'application/json',
      };

      if (body) {
        headers['Content-Type'] = 'application/json';
      }

      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await this.fetchFn(`${this.host}${path}`, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Handle error responses
      if (!response.ok) {
        let errorResponse: ErrorResponse | undefined;
        try {
          errorResponse = await response.json() as ErrorResponse;
        } catch {
          // Ignore JSON parse errors
        }

        throw new ServerError(
          errorResponse?.error ?? `HTTP ${response.status}`,
          response.status,
          errorResponse?.code,
          errorResponse?.details
        );
      }

      return await response.json() as T;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof ServerError) {
        throw error;
      }

      if ((error as Error).name === 'AbortError') {
        throw new TimeoutError(`Request timed out after ${this.timeout}ms`);
      }

      throw new ConnectionError(
        `Failed to connect to ${this.host}: ${(error as Error).message}`,
        error as Error
      );
    }
  }

  /**
   * Sleep for the specified number of milliseconds
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}