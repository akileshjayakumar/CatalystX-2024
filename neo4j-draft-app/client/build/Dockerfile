# Use the official Node.js image.
FROM node:18-alpine

# Set the working directory in the container.
WORKDIR /app

# Copy the package.json and install the dependencies.
COPY package*.json ./
RUN npm install

# Copy the rest of the client app files.
COPY . .

# Build the React app.
RUN npm run build

# Expose port 3000 and serve the app using a lightweight server.
EXPOSE 3000
CMD ["npm", "start"]
