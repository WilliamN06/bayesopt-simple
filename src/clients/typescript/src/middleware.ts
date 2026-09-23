import { RateLimiter } from './client';
import { MiddlewareOptions, CheckResult } from './types';
import { RateLimiterError } from './errors';

/**
 * Express middleware for rate limiting
 * 
 * @example
 * ```typescript
 * const limiter = new RateLimiter({ host: 'http://localhost:8080' });
 * app.use(limiter.middleware({
 *   extractClientId: (req) => req.headers['x-api-key'] as string,
 * }));
 * ```
 */
export function createMiddleware(
  limiter: RateLimiter,
  options: MiddlewareOptions
) {
  const {
    extractClientId,
    extractEndpoint = (req: any) => req.path,
    failOpen = true,
    onRateLimited,
    onError,
  } = options;

  return async (req: any, res: any, next: (err?: Error) => void) => {
    try {
      // Extract client ID
      const clientId = extractClientId(req);
      if (!clientId) {
        // If we can't identify the client, either fail open or return 401
        if (failOpen) {
          return next();
        }
        return res.status(401).json({
          error: 'Client ID is required',
          code: 'MISSING_CLIENT_ID',
        });
      }

      // Extract endpoint
      const endpoint = extractEndpoint(req);

      // Check rate limit
      const result = await limiter.check({ clientId, endpoint });

      // Set rate limit headers
      res.set('X-RateLimit-Limit', String(result.limit));
      res.set('X-RateLimit-Remaining', String(result.remaining));
      res.set('X-RateLimit-Reset', String(Math.floor(result.resetAt.getTime() / 1000)));

      if (!result.allowed) {
        res.set('Retry-After', String(result.retryAfter ?? 1));

        if (onRateLimited) {
          return onRateLimited(req, res, result);
        }

        return res.status(429).json({
          error: 'Rate limit exceeded',
          code: 'RATE_LIMITED',
          retryAfter: result.retryAfter,
        });
      }

      next();
    } catch (error) {
      if (onError) {
        return onError(req, res, error as Error);
      }

      if (failOpen) {
        // Log the error but let the request through
        console.error('Rate limiter error:', error);
        return next();
      }

      // Fail closed - return 503
      return res.status(503).json({
        error: 'Rate limiter unavailable',
        code: 'RATE_LIMITER_UNAVAILABLE',
        details: (error as Error).message,
      });
    }
  };
}

/**
 * Extend the RateLimiter class with middleware method
 */
declare module './client' {
  interface RateLimiter {
    middleware(options: MiddlewareOptions): ReturnType<typeof createMiddleware>;
  }
}

// Add middleware method to RateLimiter prototype
RateLimiter.prototype.middleware = function (options: MiddlewareOptions) {
  return createMiddleware(this, options);
};