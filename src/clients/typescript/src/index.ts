// Main client
export { RateLimiter } from './client';

// Middleware
export { createMiddleware } from './middleware';

// Types
export type {
  RateLimiterConfig,
  CheckParams,
  CheckResult,
  MiddlewareOptions,
  ConfigResponse,
  ErrorResponse,
} from './types';

// Errors
export {
  RateLimiterError,
  ConnectionError,
  TimeoutError,
  ServerError,
  ValidationError,
} from './errors';

// Add middleware method to RateLimiter
import { RateLimiter } from './client';
import { createMiddleware } from './middleware';
import { MiddlewareOptions } from './types';

declare module './client' {
  interface RateLimiter {
    middleware(options: MiddlewareOptions): ReturnType<typeof createMiddleware>;
  }
}

RateLimiter.prototype.middleware = function (options: MiddlewareOptions) {
  return createMiddleware(this, options);
};

// Default export
import { RateLimiter as DefaultRateLimiter } from './client';
export default DefaultRateLimiter;