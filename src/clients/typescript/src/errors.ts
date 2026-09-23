/**
 * Base error class for all rate limiter errors
 */
export class RateLimiterError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RateLimiterError';
    Object.setPrototypeOf(this, RateLimiterError.prototype);
  }
}

/**
 * Error thrown when the rate limiter service cannot be reached
 */
export class ConnectionError extends RateLimiterError {
  constructor(message: string, public readonly cause?: Error) {
    super(message);
    this.name = 'ConnectionError';
    Object.setPrototypeOf(this, ConnectionError.prototype);
  }
}

/**
 * Error thrown when a request times out
 */
export class TimeoutError extends RateLimiterError {
  constructor(message: string) {
    super(message);
    this.name = 'TimeoutError';
    Object.setPrototypeOf(this, TimeoutError.prototype);
  }
}

/**
 * Error thrown when the server returns an error response
 */
export class ServerError extends RateLimiterError {
  constructor(
    message: string,
    public readonly statusCode: number,
    public readonly code?: string,
    public readonly details?: string
  ) {
    super(message);
    this.name = 'ServerError';
    Object.setPrototypeOf(this, ServerError.prototype);
  }
}

/**
 * Error thrown when the request is invalid
 */
export class ValidationError extends RateLimiterError {
  constructor(message: string, public readonly field?: string) {
    super(message);
    this.name = 'ValidationError';
    Object.setPrototypeOf(this, ValidationError.prototype);
  }
}